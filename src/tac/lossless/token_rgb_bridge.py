from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMMAVQ_ROOT = PROJECT_ROOT / "workspace" / "upstream" / "commavq"
OFFICIAL_DECODER_URL = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for the commavq token->RGB bridge") from exc
    return torch


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


def load_official_commavq_bridge(
    *,
    device: str = "auto",
    dtype: str = "auto",
    commavq_root: str | Path | None = None,
    decoder_url: str = OFFICIAL_DECODER_URL,
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
    return decoder, canonical_transpose_and_clip


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
    flattened = arr.reshape(arr.shape[0], -1)

    try:
        torch = _require_torch()
    except ImportError:
        torch = None

    outputs: list[np.ndarray] = []
    for start in range(0, flattened.shape[0], batch_size):
        batch = flattened[start : start + batch_size]
        if torch is not None:
            batch_input = torch.from_numpy(np.array(batch, copy=True)).to(device=device, dtype=torch.long)
            with torch.inference_mode():
                decoded = decoder(batch_input)
            if hasattr(decoded, "detach"):
                decoded_np = decoded.detach().cpu().numpy()
            else:
                decoded_np = np.asarray(decoded)
        else:
            decoded_np = np.asarray(decoder(batch))
        outputs.append(decoded_np)
    stacked = np.concatenate(outputs, axis=0)
    return np.asarray(transpose_and_clip_fn(stacked), dtype=np.uint8)


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
    decoder, transpose_and_clip_fn = bridge_loader(
        device=device,
        dtype=dtype,
        commavq_root=commavq_root,
        decoder_url=decoder_url,
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
    return {
        "command": "lossless_token_rgb_sample",
        "token_path": str(source),
        "output_path": str(target),
        "frame_count": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "batch_size": batch_size,
        "device": device,
        "requested_dtype": dtype,
        "dtype": resolve_bridge_dtype_name(device=device, dtype=dtype),
        "commavq_root": str(resolve_commavq_official_root(commavq_root)),
        "decoder_url": decoder_url,
        "measured": False,
        "local_only": True,
    }
