from __future__ import annotations

import io
import json
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import (
    COMMAVQ_CHALLENGE_FILES,
    build_token_records,
    load_local_commavq_record_sample,
    resolve_local_commavq_cached_data_files,
)
from .profiles import TinyFramePredictorProfileConfig, load_tiny_frame_predictor_profile
from .tiny_frame_predictor import TinyFramePredictorConfig, build_tiny_frame_predictor


@dataclass(frozen=True)
class TinyFrameSupervisedBatch:
    context_frames: object
    target_frames: object
    file_names: tuple[str, ...]
    target_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        contexts = np.asarray(self.context_frames, dtype=np.int64)
        targets = np.asarray(self.target_frames, dtype=np.int64)
        file_names = tuple(str(name) for name in self.file_names)
        target_indices = tuple(int(index) for index in self.target_frame_indices)

        if contexts.ndim != 4 or contexts.shape[2:] != (8, 16):
            raise ValueError("context_frames must have shape (N, context_frames, 8, 16)")
        if targets.ndim != 3 or targets.shape[1:] != (8, 16):
            raise ValueError("target_frames must have shape (N, 8, 16)")
        if contexts.shape[0] != targets.shape[0]:
            raise ValueError("context_frames and target_frames must agree on batch size")
        if contexts.shape[0] != len(file_names) or contexts.shape[0] != len(target_indices):
            raise ValueError("batch metadata must have one entry per sample")
        if contexts.shape[0] < 1:
            raise ValueError("batch must contain at least one sample")
        if any(index <= 0 for index in target_indices):
            raise ValueError("target_frame_indices must be positive")

        object.__setattr__(self, "context_frames", contexts)
        object.__setattr__(self, "target_frames", targets)
        object.__setattr__(self, "file_names", file_names)
        object.__setattr__(self, "target_frame_indices", target_indices)


def _resolve_training_config(
    config_or_profile: TinyFramePredictorConfig | TinyFramePredictorProfileConfig | str,
    *,
    context_frames: int | None = None,
    vocab_size: int | None = None,
) -> TinyFramePredictorConfig:
    if isinstance(config_or_profile, str):
        profile = load_tiny_frame_predictor_profile(config_or_profile)
        return TinyFramePredictorConfig(
            context_frames=profile.context_frames if context_frames is None else int(context_frames),
            positions=profile.positions,
            vocab_size=profile.vocab_size if vocab_size is None else int(vocab_size),
            embed_dim=profile.embed_dim,
            hidden_dim=profile.hidden_dim,
            mixer_layers=profile.mixer_layers,
        )
    if isinstance(config_or_profile, TinyFramePredictorProfileConfig):
        return TinyFramePredictorConfig(
            context_frames=config_or_profile.context_frames if context_frames is None else int(context_frames),
            positions=config_or_profile.positions,
            vocab_size=config_or_profile.vocab_size if vocab_size is None else int(vocab_size),
            embed_dim=config_or_profile.embed_dim,
            hidden_dim=config_or_profile.hidden_dim,
            mixer_layers=config_or_profile.mixer_layers,
        )
    return TinyFramePredictorConfig(
        context_frames=config_or_profile.context_frames if context_frames is None else int(context_frames),
        positions=config_or_profile.positions,
        vocab_size=config_or_profile.vocab_size if vocab_size is None else int(vocab_size),
        embed_dim=config_or_profile.embed_dim,
        hidden_dim=config_or_profile.hidden_dim,
        mixer_layers=config_or_profile.mixer_layers,
    )


def build_tiny_frame_training_model(
    config_or_profile: TinyFramePredictorConfig | TinyFramePredictorProfileConfig | str = "tiny_frame_predictor_small",
    *,
    context_frames: int | None = None,
    vocab_size: int | None = None,
    device: str = "cpu",
):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for tiny frame training") from exc

    config = _resolve_training_config(config_or_profile, context_frames=context_frames, vocab_size=vocab_size)
    model = build_tiny_frame_predictor(config)
    return model.to(device=torch.device(device))


def _resolve_shard_paths(
    *,
    shard_paths: Sequence[str | Path] | None,
    data_files: Sequence[str] | None,
) -> list[str]:
    if shard_paths is not None:
        resolved = [str(Path(path)) for path in shard_paths]
        if not resolved:
            raise ValueError("shard_paths must not be empty")
        missing = [path for path in resolved if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"commavq shard not found: {missing[0]}")
        return resolved

    data_file_names = list(COMMAVQ_CHALLENGE_FILES if data_files is None else data_files)
    if not data_file_names:
        raise ValueError("data_files must not be empty")
    resolved = resolve_local_commavq_cached_data_files(data_file_names)
    if resolved is None:
        joined = ", ".join(data_file_names)
        raise FileNotFoundError(f"cached commavq shards not found locally: {joined}")
    return resolved


def _iter_record_supervised_samples(
    records,
    *,
    context_frames: int,
):
    for record in records:
        frames = np.asarray(record.tokens, dtype=np.int64)
        if frames.ndim != 3 or frames.shape[1:] != (8, 16):
            raise ValueError(f"record {record.file_name} must have shape (N, 8, 16)")
        if frames.shape[0] < 2:
            continue

        for target_index in range(1, int(frames.shape[0])):
            context = np.zeros((context_frames, 8, 16), dtype=np.int64)
            prefix = frames[max(0, target_index - context_frames) : target_index]
            context[-prefix.shape[0] :] = prefix
            yield context, frames[target_index], record.file_name, target_index


def iter_local_commavq_tiny_frame_batches(
    *,
    batch_size: int,
    context_frames: int,
    max_records: int,
    shard_paths: Sequence[str | Path] | None = None,
    data_files: Sequence[str] | None = None,
    sample_offset: int = 0,
    max_batches: int | None = None,
) -> Iterator[TinyFrameSupervisedBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")

    resolved_shards = _resolve_shard_paths(shard_paths=shard_paths, data_files=data_files)
    examples = load_local_commavq_record_sample(resolved_shards, max_records=max_records)
    records = build_token_records(examples)
    samples = list(_iter_record_supervised_samples(records, context_frames=context_frames))
    if not samples:
        raise ValueError("cached shards did not produce any next-frame samples")
    if sample_offset >= len(samples):
        raise ValueError("sample_offset exceeds available next-frame samples")

    yielded = 0
    for start in range(sample_offset, len(samples), batch_size):
        if max_batches is not None and yielded >= max_batches:
            break
        chunk = samples[start : start + batch_size]
        if not chunk:
            break
        yield TinyFrameSupervisedBatch(
            context_frames=np.stack([sample[0] for sample in chunk], axis=0),
            target_frames=np.stack([sample[1] for sample in chunk], axis=0),
            file_names=tuple(sample[2] for sample in chunk),
            target_frame_indices=tuple(int(sample[3]) for sample in chunk),
        )
        yielded += 1


def run_tiny_frame_supervised_step(
    model,
    batch: TinyFrameSupervisedBatch,
    *,
    optimizer=None,
    device: str = "cpu",
) -> dict[str, object]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch is required for tiny frame training") from exc

    device_obj = torch.device(device)
    model = model.to(device=device_obj)
    inputs = torch.as_tensor(
        batch.context_frames.reshape(batch.context_frames.shape[0], batch.context_frames.shape[1], -1),
        dtype=torch.long,
        device=device_obj,
    )
    targets = torch.as_tensor(
        batch.target_frames.reshape(batch.target_frames.shape[0], -1),
        dtype=torch.long,
        device=device_obj,
    )

    if optimizer is not None:
        model.train()
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
        grad_context = nullcontext()
    else:
        model.eval()
        grad_context = torch.no_grad()

    with grad_context:
        logits = model(inputs)
        if logits.ndim != 3:
            raise ValueError("tiny frame model must return logits with shape (N, positions, vocab_size)")
        if tuple(logits.shape[:2]) != tuple(targets.shape):
            raise ValueError("tiny frame model output shape does not match target frames")
        if targets.numel() > 0 and int(targets.max().detach().item()) >= int(logits.shape[-1]):
            raise ValueError("target_frames contain token ids outside the model vocab")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        predictions = logits.argmax(dim=-1)
        correct_tokens = int((predictions == targets).sum().detach().item())
        token_count = int(targets.numel())

    if optimizer is not None:
        loss.backward()
        optimizer.step()

    return {
        "mode": "train" if optimizer is not None else "eval",
        "loss": float(loss.detach().cpu().item()),
        "token_accuracy": float(correct_tokens / token_count),
        "correct_tokens": correct_tokens,
        "token_count": token_count,
        "batch_size": int(batch.context_frames.shape[0]),
        "context_frames": int(batch.context_frames.shape[1]),
        "positions": int(targets.shape[1]),
    }


def _serialized_state_dict_byte_count(model) -> int:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for tiny frame training") from exc

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def probe_tiny_frame_training(
    *,
    profile: str = "tiny_frame_predictor_small",
    output_path: str | Path | None = None,
    shard_paths: Sequence[str | Path] | None = None,
    data_files: Sequence[str] | None = None,
    batch_size: int = 2,
    context_frames: int | None = None,
    max_records: int = 1,
    sample_offset: int = 0,
    max_batches: int = 1,
    learning_rate: float = 0.05,
    device: str = "cpu",
) -> dict[str, object]:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for tiny frame training") from exc

    config = _resolve_training_config(profile, context_frames=context_frames)
    model = build_tiny_frame_training_model(config, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))

    train_steps: list[dict[str, object]] = []
    train_summary: dict[str, object] | None = None
    last_batch: TinyFrameSupervisedBatch | None = None
    for step_index, batch in enumerate(
        iter_local_commavq_tiny_frame_batches(
            shard_paths=shard_paths,
            data_files=data_files,
            batch_size=batch_size,
            context_frames=config.context_frames,
            max_records=max_records,
            sample_offset=sample_offset,
            max_batches=max_batches,
        ),
        start=1,
    ):
        train_summary = run_tiny_frame_supervised_step(model, batch, optimizer=optimizer, device=device)
        train_steps.append({"step": step_index, **train_summary})
        last_batch = batch

    if train_summary is None or last_batch is None:
        raise ValueError("tiny frame train probe did not observe any batches")

    eval_summary = run_tiny_frame_supervised_step(model, last_batch, optimizer=None, device=device)
    payload = {
        "command": "lossless_tiny_frame_train_probe",
        "profile": profile,
        "output_path": str(Path(output_path)) if output_path is not None else None,
        "device": device,
        "batch_size": int(batch_size),
        "context_frames": int(config.context_frames),
        "max_records": int(max_records),
        "sample_offset": int(sample_offset),
        "max_batches": int(max_batches),
        "learning_rate": float(learning_rate),
        "parameter_count": int(sum(int(param.numel()) for param in model.parameters())),
        "state_dict_bytes": _serialized_state_dict_byte_count(model),
        "observed_batch_count": len(train_steps),
        "train_steps": train_steps,
        "train": train_summary,
        "eval": eval_summary,
        "file_names": list(last_batch.file_names),
        "target_frame_indices": [int(index) for index in last_batch.target_frame_indices],
    }

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
