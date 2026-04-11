from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .arithmetic import FRAME_BOS_TOKEN, SEGMENT_EOT_TOKEN, load_gpt_arithmetic_profile

OFFICIAL_COMMAVQ_GPT_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin"
OFFICIAL_COMMAVQ_GPT_BLOCK_SIZE = 20 * 129
OFFICIAL_COMMAVQ_GPT_TOKENS_PER_FRAME = 129


class NextTokenLogitsModel(Protocol):
    def next_token_logits(self, context: np.ndarray) -> np.ndarray: ...


def _resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"unsupported device: {device}")

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, *, device: str) -> str:
    normalized = dtype.strip().lower()
    if normalized in {"float32", "float16", "bfloat16"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"unsupported dtype: {dtype}")
    if device == "cuda":
        return "bfloat16"
    return "float32"


def _candidate_gpt_module_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    env_path = os.environ.get("COMMAVQ_GPT_MODULE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/tmp/commavq-read/utils/gpt.py"),
            Path.cwd() / "workspace" / "upstream" / "commavq" / "utils" / "gpt.py",
        ]
    )
    return tuple(candidates)


def _resolve_gpt_module_path(gpt_module_path: str | Path | None) -> Path:
    if gpt_module_path is not None:
        path = Path(gpt_module_path)
        if not path.is_file():
            raise FileNotFoundError(f"official commavq GPT module not found: {path}")
        return path
    for candidate in _candidate_gpt_module_paths():
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("official commavq GPT module not found; set --gpt-module-path or COMMAVQ_GPT_MODULE")


def _load_official_gpt_module(gpt_module_path: str | Path | None):
    module_path = _resolve_gpt_module_path(gpt_module_path)
    spec = importlib.util.spec_from_file_location("commavq_official_gpt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load official commavq GPT module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TorchNextTokenLogitsModel:
    def __init__(self, model, *, device: str) -> None:
        self._model = model
        self._device = device
        self._block_size = int(model.config.block_size)

    def next_token_logits(self, context: np.ndarray) -> np.ndarray:
        import torch

        idx = torch.as_tensor(context, dtype=torch.long, device=self._device).view(1, -1)
        with torch.inference_mode():
            logits = self._model(idx)
        return logits[0, -1].detach().float().cpu().numpy()

    def score_tokens(
        self,
        tokens: np.ndarray,
        *,
        context_tokens: int,
        max_scored_tokens: int | None = None,
    ) -> dict[str, object]:
        import torch

        arr = np.asarray(tokens, dtype=np.int64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("tokens must be a 1D array with at least two items")
        available = int(arr.size - 1)
        score_count = available if max_scored_tokens is None else min(max_scored_tokens, available)
        if score_count <= 0:
            raise ValueError("max_scored_tokens must allow at least one scored token")
        usable_context_tokens = min(max(int(context_tokens), 1), self._block_size - 1)

        total_nll_nats = 0.0
        total_scored_tokens = 0
        for chunk_start, chunk_end, local_target_start, predictions_to_take in _iter_score_chunks(
            token_count=arr.size,
            score_count=score_count,
            context_tokens=usable_context_tokens,
            block_size=self._block_size,
        ):
            chunk = arr[chunk_start:chunk_end]
            idx = torch.as_tensor(chunk[:-1], dtype=torch.long, device=self._device).view(1, -1)
            targets = torch.as_tensor(chunk[1:], dtype=torch.long, device=self._device)
            with torch.inference_mode():
                logits = self._model(idx)
                log_probs = torch.log_softmax(logits[0], dim=-1)
            target_slice = targets[local_target_start : local_target_start + predictions_to_take]
            row_indices = torch.arange(
                local_target_start,
                local_target_start + predictions_to_take,
                device=self._device,
            )
            step_nll = -log_probs[row_indices, target_slice].sum()
            total_nll_nats += float(step_nll.detach().float().cpu().item())
            total_scored_tokens += predictions_to_take

        return {
            "scored_tokens": total_scored_tokens,
            "avg_nll_nats": total_nll_nats / total_scored_tokens,
        }


def load_official_commavq_gpt_model(
    *,
    device: str = "auto",
    dtype: str = "auto",
    cache_dir: str | Path | None = None,
    model_url: str | None = None,
    gpt_module_path: str | Path | None = None,
) -> NextTokenLogitsModel:
    import torch

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, device=resolved_device)
    module = _load_official_gpt_module(gpt_module_path)
    config = module.GPTConfig()
    model = module.GPT(config)
    load_kwargs = {}
    if cache_dir is not None:
        load_kwargs["model_dir"] = str(cache_dir)
    model.load_state_dict_from_url(url=model_url or OFFICIAL_COMMAVQ_GPT_URL, **load_kwargs)
    torch_dtype = getattr(torch, resolved_dtype)
    if resolved_device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    model.eval()
    model.to(device=resolved_device, dtype=torch_dtype)
    return _TorchNextTokenLogitsModel(model, device=resolved_device)


def _logsumexp(logits: np.ndarray) -> float:
    max_logit = float(np.max(logits))
    return max_logit + float(np.log(np.exp(logits - max_logit).sum()))


def score_tokens_with_logits_fn(
    tokens,
    *,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    context_tokens: int,
    vocab_size: int,
    max_scored_tokens: int | None = None,
) -> dict[str, object]:
    arr = np.asarray(tokens, dtype=np.int64)
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")
    if arr.ndim != 1:
        raise ValueError("tokens must be a 1D array")
    if arr.size < 2:
        raise ValueError("tokens must contain at least two items to score")

    available = int(arr.size - 1)
    score_count = available if max_scored_tokens is None else min(max_scored_tokens, available)
    if score_count <= 0:
        raise ValueError("max_scored_tokens must allow at least one scored token")

    total_nll_nats = 0.0
    for next_index in range(1, score_count + 1):
        start = max(0, next_index - context_tokens)
        context = arr[start:next_index]
        logits = np.asarray(logits_fn(context), dtype=np.float64)
        if logits.ndim != 1:
            raise ValueError("logits_fn must return a 1D array")
        if logits.size < vocab_size:
            raise ValueError("logits_fn returned fewer logits than vocab_size")
        target = int(arr[next_index])
        if target < 0 or target >= vocab_size:
            raise ValueError(f"token {target} is outside vocab size {vocab_size}")
        total_nll_nats += _logsumexp(logits[:vocab_size]) - float(logits[target])

    avg_nll_nats = total_nll_nats / score_count
    bits_per_token = avg_nll_nats / math.log(2.0)
    payload = {
        "command": "lossless_gpt_score_sample",
        "token_count": int(arr.size),
        "scored_tokens": int(score_count),
        "context_tokens": int(context_tokens),
        "vocab_size": int(vocab_size),
        "avg_nll_nats": avg_nll_nats,
        "bits_per_token": bits_per_token,
        "perplexity": math.exp(avg_nll_nats),
        "local_only": True,
        "measured": False,
    }
    return payload


def _read_uint16_tokens(token_path: str | Path) -> np.ndarray:
    source = Path(token_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    tokens = np.fromfile(source, dtype=np.uint16)
    if tokens.size == 0:
        raise ValueError(f"token stream is empty: {source}")
    return tokens


def _load_frame_major_segments_from_path(
    token_path: str | Path,
    *,
    max_scored_tokens: int | None = None,
) -> tuple[int, list[np.ndarray]]:
    source = Path(token_path)
    raw_token_count = source.stat().st_size // 2
    tokens = np.memmap(source, dtype=np.uint16, mode="r")
    segments: list[np.ndarray] = []
    current: list[int] = []
    remaining = max_scored_tokens
    for raw_token in tokens:
        token = int(raw_token)
        if token == SEGMENT_EOT_TOKEN:
            if current:
                segment = np.asarray(current, dtype=np.uint16)
                _validate_frame_major_segment(segment)
                segments.append(segment)
                current = []
                if remaining is not None:
                    remaining -= max(0, min(int(segment.size) - 1, remaining))
                    if remaining <= 0:
                        break
            continue
        if token > FRAME_BOS_TOKEN:
            raise ValueError(f"token {token} exceeds official GPT vocab upper bound {FRAME_BOS_TOKEN}")
        current.append(token)
    if current and (remaining is None or remaining > 0):
        segment = np.asarray(current, dtype=np.uint16)
        _validate_frame_major_segment(segment)
        segments.append(segment)
    if not segments:
        raise ValueError("prepared token stream did not contain any GPT-scoreable segments")
    return raw_token_count, segments


def _iter_score_chunks(*, token_count: int, score_count: int, context_tokens: int, block_size: int):
    if token_count < 2:
        raise ValueError("token_count must be at least two")
    if score_count <= 0:
        raise ValueError("score_count must be positive")
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if block_size <= 1:
        raise ValueError("block_size must be greater than one")

    usable_context_tokens = min(context_tokens, block_size - 1)
    scored = 0
    while scored < score_count:
        next_target_index = scored + 1
        chunk_start = max(0, next_target_index - usable_context_tokens)
        chunk_end = min(token_count, chunk_start + block_size, next_target_index + (score_count - scored))
        local_target_start = scored - chunk_start
        available_predictions = (chunk_end - chunk_start - 1) - local_target_start
        predictions_to_take = min(score_count - scored, available_predictions)
        if predictions_to_take <= 0:
            raise ValueError("invalid chunk plan produced zero predictions")
        yield chunk_start, chunk_end, local_target_start, predictions_to_take
        scored += predictions_to_take


def _validate_frame_major_segment(segment: np.ndarray) -> None:
    if segment.size < OFFICIAL_COMMAVQ_GPT_TOKENS_PER_FRAME:
        return
    if segment.size % OFFICIAL_COMMAVQ_GPT_TOKENS_PER_FRAME != 0:
        raise ValueError("prepared token stream is not frame-major: segment length is not divisible by 129")
    frames = segment.reshape(-1, OFFICIAL_COMMAVQ_GPT_TOKENS_PER_FRAME)
    if not np.all(frames[:, 0] == FRAME_BOS_TOKEN):
        raise ValueError("prepared token stream is not frame-major: missing BOS cadence at frame boundaries")
    if np.any(frames[:, 1:] == FRAME_BOS_TOKEN):
        raise ValueError("prepared token stream is not frame-major: BOS token appears inside frame payload")


def _split_frame_major_segments_for_official_gpt(tokens: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    current: list[int] = []
    for token in tokens.tolist():
        if token == SEGMENT_EOT_TOKEN:
            if current:
                segment = np.asarray(current, dtype=np.uint16)
                _validate_frame_major_segment(segment)
                segments.append(segment)
                current = []
            continue
        if token > FRAME_BOS_TOKEN:
            raise ValueError(f"token {token} exceeds official GPT vocab upper bound {FRAME_BOS_TOKEN}")
        current.append(token)
    if current:
        segment = np.asarray(current, dtype=np.uint16)
        _validate_frame_major_segment(segment)
        segments.append(segment)
    if not segments:
        raise ValueError("prepared token stream did not contain any GPT-scoreable segments")
    return segments


def score_commavq_gpt_sample(
    token_path: str | Path,
    *,
    output_path: str | Path | None = None,
    profile: str = "gpt_arithmetic_small",
    max_scored_tokens: int | None = None,
    context_tokens: int | None = None,
    vocab_size: int = FRAME_BOS_TOKEN + 1,
    device: str = "auto",
    dtype: str = "auto",
    cache_dir: str | Path | None = None,
    model_url: str | None = None,
    gpt_module_path: str | Path | None = None,
    model_loader: Callable[..., NextTokenLogitsModel] | None = None,
) -> dict[str, object]:
    _ = load_gpt_arithmetic_profile(profile)
    effective_context_tokens = context_tokens if context_tokens is not None else OFFICIAL_COMMAVQ_GPT_BLOCK_SIZE
    if effective_context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, device=resolved_device)
    loader = model_loader or load_official_commavq_gpt_model
    load_kwargs = {
        "device": resolved_device,
        "dtype": resolved_dtype,
        "cache_dir": cache_dir,
        "model_url": model_url,
    }
    if model_loader is None:
        load_kwargs["gpt_module_path"] = gpt_module_path
    model = loader(**load_kwargs)
    raw_token_count, segments = _load_frame_major_segments_from_path(
        token_path,
        max_scored_tokens=max_scored_tokens,
    )

    remaining = max_scored_tokens
    total_scored_tokens = 0
    total_nll_nats = 0.0
    scoreable_segments = 0
    segment_token_count = 0
    for segment in segments:
        if segment.size < 2:
            continue
        if remaining is not None and remaining <= 0:
            break
        segment_limit = remaining
        if hasattr(model, "score_tokens"):
            segment_result = model.score_tokens(
                segment,
                context_tokens=effective_context_tokens,
                max_scored_tokens=segment_limit,
            )
        else:
            segment_result = score_tokens_with_logits_fn(
                segment,
                logits_fn=model.next_token_logits,
                context_tokens=effective_context_tokens,
                vocab_size=vocab_size,
                max_scored_tokens=segment_limit,
            )
        total_scored_tokens += int(segment_result["scored_tokens"])
        total_nll_nats += float(segment_result["avg_nll_nats"]) * int(segment_result["scored_tokens"])
        scoreable_segments += 1
        segment_token_count += min(int(segment.size), int(segment_result["scored_tokens"]) + 1)
        if remaining is not None:
            remaining -= int(segment_result["scored_tokens"])

    if total_scored_tokens <= 0:
        raise ValueError("prepared token stream did not contain enough tokens to score")

    avg_nll_nats = total_nll_nats / total_scored_tokens
    payload = {
        "command": "lossless_gpt_score_sample",
        "profile": profile,
        "token_path": str(Path(token_path)),
        "output_path": str(Path(output_path)) if output_path is not None else None,
        "layout": "frame_major",
        "segment_count": len(segments),
        "scored_segment_count": scoreable_segments,
        "segment_token_count": segment_token_count,
        "raw_token_count": int(raw_token_count),
        "scored_tokens": total_scored_tokens,
        "context_tokens": int(effective_context_tokens),
        "vocab_size": int(vocab_size),
        "device": resolved_device,
        "dtype": resolved_dtype,
        "model_url": model_url or OFFICIAL_COMMAVQ_GPT_URL,
        "model_block_size": OFFICIAL_COMMAVQ_GPT_BLOCK_SIZE,
        "avg_nll_nats": avg_nll_nats,
        "bits_per_token": avg_nll_nats / math.log(2.0),
        "perplexity": math.exp(avg_nll_nats),
        "local_only": True,
        "measured": False,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
