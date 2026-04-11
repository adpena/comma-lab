from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMMAVQ_ROOT = PROJECT_ROOT / "workspace" / "upstream" / "commavq"
OFFICIAL_TORCH_DECODER_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin"
OFFICIAL_ONNX_DECODER_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder.onnx"
OFFICIAL_DECODER_URL = OFFICIAL_TORCH_DECODER_URL


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for the commavq token->RGB bridge") from exc
    return torch


def _require_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError("onnxruntime is required for the official ONNX commavq decoder bridge") from exc
    return onnxruntime


def canonical_transpose_and_clip(tensors) -> np.ndarray:
    arr = np.array(tensors)
    arr = np.transpose(arr, (0, 2, 3, 1))
    return np.clip(arr, 0, 255).astype(np.uint8)


def resolve_bridge_dtype_name(*, device: str, dtype: str) -> str:
    # The official decoder quantizer path builds float32 one-hot encodings,
    # so half precision is not a safe runtime default without patching the
    # canonical model implementation.
    if dtype in {"auto", "float16", "bfloat16"}:
        return "float32"
    return dtype


def resolve_onnx_execution_providers(available_providers: Sequence[str]) -> list[str]:
    preferred = [
        provider
        for provider in ("CoreMLExecutionProvider", "CPUExecutionProvider")
        if provider in set(available_providers)
    ]
    if not preferred:
        raise RuntimeError(
            "official ONNX commavq decoder requires CoreMLExecutionProvider or CPUExecutionProvider"
        )
    return preferred


def _default_decoder_onnx_path() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache_home / "tac" / "commavq-gpt2m" / "decoder.onnx"


def ensure_official_decoder_onnx_path(*, decoder_url: str = OFFICIAL_ONNX_DECODER_URL) -> Path:
    env_path = os.environ.get("TAC_COMMAVQ_DECODER_ONNX")
    if env_path:
        candidate = Path(env_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"TAC_COMMAVQ_DECODER_ONNX does not point to a file: {candidate}")
        return candidate

    target = _default_decoder_onnx_path()
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(decoder_url) as response, tmp_target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_target.replace(target)
    finally:
        if tmp_target.exists():
            tmp_target.unlink()
    return target


def resolve_commavq_official_root(commavq_root: str | Path | None = None) -> Path:
    candidates = []
    if commavq_root is not None:
        candidates.append(Path(commavq_root))
    env_root = os.environ.get("COMMAVQ_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(DEFAULT_COMMAVQ_ROOT)

    required = [
        Path("utils/vqvae.py"),
        Path("notebooks/decode.ipynb"),
    ]
    for root in candidates:
        if root.exists() and all((root / rel).exists() for rel in required):
            return root
    raise FileNotFoundError(
        "Could not locate canonical commaai/commavq checkout with utils/vqvae.py and notebooks/decode.ipynb"
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _OnnxDecoderBridge:
    _tac_input_kind = "numpy"

    def __init__(self, session) -> None:
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("official ONNX commavq decoder session has no inputs")
        self._session = session
        self._input_name = inputs[0].name

    def __call__(self, batch) -> np.ndarray:
        if hasattr(batch, "detach"):
            batch = batch.detach().cpu().numpy()
        tokens = np.asarray(batch, dtype=np.int64)
        outputs = self._session.run(None, {self._input_name: tokens})
        if not outputs:
            raise RuntimeError("official ONNX commavq decoder session returned no outputs")
        return np.asarray(outputs[0])


def load_official_commavq_onnx_bridge(*, decoder_url: str = OFFICIAL_ONNX_DECODER_URL):
    onnxruntime = _require_onnxruntime()
    decoder_path = ensure_official_decoder_onnx_path(decoder_url=decoder_url)
    providers = resolve_onnx_execution_providers(onnxruntime.get_available_providers())
    session = onnxruntime.InferenceSession(str(decoder_path), providers=providers)
    return _OnnxDecoderBridge(session), canonical_transpose_and_clip, {
        "bridge_backend": "onnx",
        "execution_provider": providers[0],
        "execution_providers": providers,
        "decoder_artifact_url": decoder_url,
        "decoder_artifact_path": str(decoder_path),
    }


def load_official_commavq_torch_bridge(
    *,
    device: str = "auto",
    dtype: str = "auto",
    commavq_root: str | Path | None = None,
    decoder_url: str = OFFICIAL_TORCH_DECODER_URL,
):
    root = resolve_commavq_official_root(commavq_root)
    vqvae = _load_module("commavq_utils_vqvae", root / "utils" / "vqvae.py")
    torch = _require_torch()

    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    resolved_dtype_name = resolve_bridge_dtype_name(device=resolved_device, dtype=dtype)
    resolved_dtype = getattr(torch, resolved_dtype_name)

    config = vqvae.CompressorConfig()
    with torch.device("meta"):
        decoder = vqvae.Decoder(config)
    decoder.load_state_dict_from_url(decoder_url, assign=True)
    decoder = decoder.eval().to(device=resolved_device, dtype=resolved_dtype)
    return decoder, canonical_transpose_and_clip, {
        "bridge_backend": "torch",
        "decoder_artifact_url": decoder_url,
        "commavq_root": str(root),
    }


def _resolve_bridge_decoder_urls(decoder_url: str) -> tuple[str, str]:
    if decoder_url.endswith(".onnx"):
        return decoder_url, OFFICIAL_TORCH_DECODER_URL
    return OFFICIAL_ONNX_DECODER_URL, decoder_url


def load_official_commavq_bridge(
    *,
    device: str = "auto",
    dtype: str = "auto",
    commavq_root: str | Path | None = None,
    decoder_url: str = OFFICIAL_DECODER_URL,
):
    onnx_decoder_url, torch_decoder_url = _resolve_bridge_decoder_urls(decoder_url)
    try:
        return load_official_commavq_onnx_bridge(decoder_url=onnx_decoder_url)
    except Exception as exc:
        decoder, transpose_and_clip_fn, metadata = load_official_commavq_torch_bridge(
            device=device,
            dtype=dtype,
            commavq_root=commavq_root,
            decoder_url=torch_decoder_url,
        )
        metadata = dict(metadata)
        metadata["bridge_fallback_reason"] = str(exc)
        return decoder, transpose_and_clip_fn, metadata


def _normalize_token_cube(tokens) -> np.ndarray:
    arr = np.asarray(tokens)
    if arr.ndim == 2 and arr.shape[1] == 128:
        return arr.astype(np.int64, copy=False).reshape(arr.shape[0], 8, 16)
    if arr.ndim != 3 or tuple(arr.shape[1:]) != (8, 16):
        raise ValueError("token array must have shape (frames, 128) or (frames, 8, 16)")
    return arr.astype(np.int64, copy=False)


def decode_commavq_tokens_to_rgb_frames(
    tokens,
    *,
    decoder,
    transpose_and_clip_fn: Callable[[np.ndarray], np.ndarray],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    arr = _normalize_token_cube(tokens)

    if getattr(decoder, "_tac_input_kind", "torch") == "numpy":
        torch = None
        numpy_input_kind = "cube"
    else:
        numpy_input_kind = "flat"
        try:
            torch = _require_torch()
        except ImportError:
            torch = None

    outputs: list[np.ndarray] = []
    for start in range(0, arr.shape[0], batch_size):
        cube_batch = arr[start : start + batch_size]
        if torch is not None:
            batch_input = torch.from_numpy(np.array(cube_batch.reshape(cube_batch.shape[0], -1), copy=True)).to(
                device=device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                decoded = decoder(batch_input)
            if hasattr(decoded, "detach"):
                decoded_np = decoded.detach().cpu().numpy()
            else:
                decoded_np = np.asarray(decoded)
        else:
            if numpy_input_kind == "cube":
                decoded_np = np.asarray(decoder(np.array(cube_batch, copy=False)))
            else:
                decoded_np = np.asarray(decoder(np.array(cube_batch.reshape(cube_batch.shape[0], -1), copy=True)))
        outputs.append(decoded_np)
    stacked = np.concatenate(outputs, axis=0)
    return np.asarray(transpose_and_clip_fn(stacked), dtype=np.uint8)


def _unpack_bridge_loader_result(result):
    if not isinstance(result, tuple):
        raise TypeError("bridge loader must return a tuple")
    if len(result) == 2:
        decoder, transpose_and_clip_fn = result
        return decoder, transpose_and_clip_fn, {}
    if len(result) == 3:
        decoder, transpose_and_clip_fn, metadata = result
        return decoder, transpose_and_clip_fn, dict(metadata or {})
    raise ValueError("bridge loader must return (decoder, transpose_and_clip_fn[, metadata])")


def decode_commavq_token_file_to_rgb(
    *,
    token_path: str | Path,
    output_path: str | Path,
    max_frames: int | None = None,
    batch_size: int = 64,
    device: str = "auto",
    dtype: str = "auto",
    commavq_root: str | Path | None = None,
    decoder_url: str = OFFICIAL_DECODER_URL,
    bridge_loader=load_official_commavq_bridge,
) -> dict[str, object]:
    source = Path(token_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tokens = np.load(source, mmap_mode="r", allow_pickle=False)
    token_cube = _normalize_token_cube(tokens)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        token_cube = token_cube[:max_frames]
    decoder, transpose_and_clip_fn, bridge_metadata = _unpack_bridge_loader_result(
        bridge_loader(
            device=device,
            dtype=dtype,
            commavq_root=commavq_root,
            decoder_url=decoder_url,
        )
    )
    frames = decode_commavq_tokens_to_rgb_frames(
        token_cube,
        decoder=decoder,
        transpose_and_clip_fn=transpose_and_clip_fn,
        batch_size=batch_size,
        device=("cpu" if device == "auto" else device),
    )
    memmap = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint8, shape=frames.shape)
    memmap[:] = frames
    del memmap

    resolved_commavq_root = bridge_metadata.get("commavq_root")
    if resolved_commavq_root is None and commavq_root is not None:
        resolved_commavq_root = str(resolve_commavq_official_root(commavq_root))
    actual_decoder_url = bridge_metadata.get("decoder_artifact_url", decoder_url)

    result = {
        "command": "lossless_token_rgb_sample",
        "token_path": str(source),
        "output_path": str(target),
        "frame_count": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "batch_size": batch_size,
        "device": device,
        "requested_dtype": dtype,
        "dtype": resolve_bridge_dtype_name(device=device, dtype=dtype),
        "commavq_root": resolved_commavq_root,
        "decoder_url": actual_decoder_url,
        "measured": False,
        "local_only": True,
    }
    result.update(bridge_metadata)
    return result
