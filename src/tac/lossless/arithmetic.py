from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Optional

from .data import COMMAVQ_DATASET_NAME, load_commavq_dataset
from .profiles import PROFILES

FRAME_BOS_TOKEN = 1024
SEGMENT_EOT_TOKEN = 1025


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
    split: tuple[str, ...]
    work_dir: str | None
    status: str = "planned"
    measured: bool = False

    def __post_init__(self) -> None:
        if self.method != "gpt_arithmetic":
            raise ValueError(f"unsupported arithmetic method: {self.method}")
        if not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
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


def flatten_tokens_for_gpt_arithmetic(tokens, *, bos_token: int = FRAME_BOS_TOKEN, eot_token: int = SEGMENT_EOT_TOKEN):
    import numpy as np

    array = np.asarray(tokens).astype(np.int16)
    if array.ndim < 2:
        raise ValueError("tokens must have at least frame and feature dimensions")
    frames = array.shape[0]
    flat_per_frame = array.reshape(frames, -1)
    bos = np.full((frames, 1), bos_token, dtype=np.int16)
    with_bos = np.concatenate([bos, flat_per_frame], axis=1)
    flattened = with_bos.reshape(-1)
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


def estimate_gpt_arithmetic_workload(
    profile: str,
    *,
    split: str | Sequence[str] | None = "challenge",
    work_dir: str | Path | None = None,
    dataset_loader=None,
    num_proc: Optional[int] = None,
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
    tokens_per_frame = raw_tokens_per_frame + 1
    flat_tokens_per_example = frames_per_example * tokens_per_frame + 1
    total_flat_tokens = example_count * flat_tokens_per_example
    return GPTArithmeticEstimate(
        profile=config.profile,
        method=config.method,
        model=config.model,
        context_tokens=config.context_tokens,
        dataset_name=config.dataset_name,
        split=_normalize_split(split),
        work_dir=str(work_dir) if work_dir is not None else None,
        example_count=example_count,
        frames_per_example=frames_per_example,
        tokens_per_frame=tokens_per_frame,
        flat_tokens_per_example=flat_tokens_per_example,
        total_flat_tokens=total_flat_tokens,
    )


def build_gpt_arithmetic_plan(
    profile: str,
    *,
    split: str | Sequence[str] | None = "challenge",
    work_dir: str | Path | None = None,
) -> GPTArithmeticPlan:
    config = load_gpt_arithmetic_profile(profile)
    return GPTArithmeticPlan(
        profile=config.profile,
        method=config.method,
        model=config.model,
        context_tokens=config.context_tokens,
        dataset_name=config.dataset_name,
        split=_normalize_split(split),
        work_dir=str(work_dir) if work_dir is not None else None,
    )
