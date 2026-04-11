from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

BytesLike = bytes | bytearray | memoryview
TokenSource = BytesLike | Iterable[int]

COMMAVQ_DATASET_NAME = "commaai/commavq"
COMMAVQ_CHALLENGE_FILES = ("data-0000.tar.gz", "data-0001.tar.gz")
COMMAVQ_TOKENS_PER_EXAMPLE = 1200 * 128
COMMAVQ_TOKEN_BITS = 10


def _normalize_data_file_entry(value: object) -> str:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("split indices must be non-negative")
        return f"data-{value:04d}.tar.gz"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("split entries must be non-empty")
        if text.isdigit():
            return f"data-{int(text):04d}.tar.gz"
        return text
    raise ValueError(f"unsupported split entry: {value!r}")


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for commavq lossless token array helpers") from exc
    return np


@dataclass(frozen=True)
class TokenRecord:
    file_name: str
    tokens: object

    def __post_init__(self) -> None:
        if not isinstance(self.file_name, str) or not self.file_name.strip():
            raise ValueError("file_name must be a non-empty string")


def token_bytes(tokens: TokenSource) -> bytes:
    if isinstance(tokens, bytes):
        return tokens
    if isinstance(tokens, bytearray):
        return bytes(tokens)
    if isinstance(tokens, memoryview):
        return tokens.tobytes()

    try:
        values = list(tokens)
    except TypeError as exc:
        raise TypeError("tokens must be bytes-like or an iterable of integers") from exc

    normalized = bytearray()
    for index, value in enumerate(values):
        if not isinstance(value, int):
            raise TypeError(f"token at index {index} must be an integer")
        if value < 0 or value > 255:
            raise ValueError(f"token at index {index} must be in range 0..255")
        normalized.append(value)
    return bytes(normalized)


def token_byte_length(tokens: TokenSource) -> int:
    return len(token_bytes(tokens))


def resolve_commavq_data_files(
    split: str | Sequence[str] | Mapping[str, Sequence[str]] | None = "challenge",
) -> dict[str, list[str]]:
    if split is None or split == "challenge" or split == "train":
        return {"train": list(COMMAVQ_CHALLENGE_FILES)}

    if isinstance(split, str):
        raise ValueError(f"Unknown commavq split alias: {split}")

    if isinstance(split, Mapping):
        resolved: dict[str, list[str]] = {}
        for key, values in split.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("split keys must be non-empty strings")
            resolved[key] = [_normalize_data_file_entry(value) for value in values]
        return resolved

    return {"train": [_normalize_data_file_entry(value) for value in split]}


def commavq_original_bytes(
    example_count: int,
    *,
    tokens_per_example: int = COMMAVQ_TOKENS_PER_EXAMPLE,
    bits_per_token: int = COMMAVQ_TOKEN_BITS,
) -> int:
    """Compute the uncompressed byte size of *example_count* commavq examples.

    This is the *theoretical* original size used as denominator for
    compression-rate calculations, not a filesystem size.  Each example
    contains ``tokens_per_example`` tokens of ``bits_per_token`` bits.
    """
    if example_count < 0:
        raise ValueError("example_count must be non-negative")
    if tokens_per_example <= 0:
        raise ValueError("tokens_per_example must be positive")
    if bits_per_token <= 0:
        raise ValueError("bits_per_token must be positive")
    return example_count * tokens_per_example * bits_per_token // 8


def normalize_token_array(tokens: object):
    np = _require_numpy()
    return np.asarray(tokens)


def load_token_file(path):
    """Load a token .npy file and validate it contains a numeric array."""
    np = _require_numpy()
    arr = np.load(path, allow_pickle=False)
    if not hasattr(arr, "dtype") or arr.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"token file {path} does not contain a numeric array")
    if arr.ndim == 0:
        raise ValueError(f"token file {path} contains a scalar, expected an array")
    return arr


def build_token_records(examples: Iterable[Mapping[str, object]]) -> list[TokenRecord]:
    records: list[TokenRecord] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"example {index} must be a mapping")
        meta = example.get("json")
        if not isinstance(meta, Mapping):
            raise ValueError(f"example {index} is missing json metadata")
        file_name = meta.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"example {index} is missing json.file_name")
        if "token.npy" not in example:
            raise ValueError(f"example {index} is missing token.npy")
        records.append(TokenRecord(file_name=file_name, tokens=normalize_token_array(example["token.npy"])))
    return sorted(records, key=lambda record: record.file_name)


def load_commavq_reference_records(
    *,
    split: str | Sequence[str] | Mapping[str, Sequence[str]] | None = "challenge",
    dataset_loader=None,
    dataset_name: str = COMMAVQ_DATASET_NAME,
    num_proc: int | None = None,
) -> list[TokenRecord]:
    dataset = load_commavq_dataset(
        split=split,
        dataset_loader=dataset_loader,
        dataset_name=dataset_name,
        num_proc=num_proc,
    )

    if isinstance(dataset, Mapping):
        split_examples = []
        for values in dataset.values():
            split_examples.extend(list(values))
        return build_token_records(split_examples)

    raise ValueError("dataset_loader must return a mapping of splits to example iterables")


def load_commavq_dataset(
    *,
    split: str | Sequence[str] | Mapping[str, Sequence[str]] | None = "challenge",
    dataset_loader=None,
    dataset_name: str = COMMAVQ_DATASET_NAME,
    num_proc: int | None = None,
):
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    return dataset_loader(
        dataset_name,
        num_proc=num_proc,
        data_files=resolve_commavq_data_files(split),
    )
