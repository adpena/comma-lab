from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .profiles import TinyFramePredictorProfileConfig, load_tiny_frame_predictor_profile


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("torch is required for tiny frame predictor experiments") from exc
    return torch, nn


@dataclass(frozen=True)
class TinyFramePredictorConfig:
    context_frames: int
    positions: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    mixer_layers: int

    def __post_init__(self) -> None:
        if self.context_frames <= 0:
            raise ValueError("context_frames must be positive")
        if self.positions <= 0:
            raise ValueError("positions must be positive")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if self.embed_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")
        if self.mixer_layers <= 0:
            raise ValueError("mixer_layers must be positive")


class TinyFramePredictor:
    def __init__(
        self,
        model,
        *,
        config: TinyFramePredictorConfig,
        profile: str,
        torch,
        device: str,
        dtype: str,
        fallback_reason: str | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._tac_model_backend = "tiny_frame_predictor"
        self._tac_model_profile = profile
        self._tac_execution_provider = device
        self._tac_execution_providers = [device]
        if fallback_reason:
            self._tac_bridge_fallback_reason = fallback_reason

    def _prepare_tokens(self, prefix_frames: np.ndarray, *, context_frames: int) -> np.ndarray:
        if int(context_frames) != self._config.context_frames:
            raise ValueError("context_frames does not match tiny frame predictor runtime")
        arr = np.asarray(prefix_frames, dtype=np.uint16)
        if arr.ndim != 3 or arr.shape[1:] != (8, 16):
            raise ValueError("prefix_frames must have shape (N, 8, 16)")
        if arr.shape[0] < 1:
            raise ValueError("prefix_frames must contain at least one frame")
        flat = arr.reshape(arr.shape[0], -1)
        if flat.shape[1] != self._config.positions:
            raise ValueError("prefix_frames positions do not match runtime config")
        if int(flat.max()) >= self._config.vocab_size:
            raise ValueError("prefix_frames contain token ids outside the runtime vocab")

        padded = np.zeros((self._config.context_frames, self._config.positions), dtype=np.int64)
        usable = flat[-self._config.context_frames :]
        padded[-usable.shape[0] :] = usable.astype(np.int64, copy=False)
        return padded

    def next_frame_logits(self, prefix_frames: np.ndarray, *, context_frames: int) -> np.ndarray:
        torch = self._torch
        tokens = self._prepare_tokens(prefix_frames, context_frames=context_frames)
        with torch.no_grad():
            token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=self._device).unsqueeze(0)
            logits = self._model(token_tensor)
        return logits[0].detach().to("cpu", dtype=torch.float32).numpy().astype(np.float64, copy=False)


def build_tiny_frame_predictor(config: TinyFramePredictorConfig):
    torch, nn = _require_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        def forward(self, x):
            return x + self.ff(self.norm(x))

    class _TinyFramePredictor(nn.Module):
        def __init__(self, cfg: TinyFramePredictorConfig) -> None:
            super().__init__()
            self.config = cfg
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
            self.position_embedding = nn.Parameter(torch.zeros(cfg.positions, cfg.embed_dim))
            self.frame_embedding = nn.Parameter(torch.zeros(cfg.context_frames, cfg.embed_dim))
            self.frame_projection = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
            self.mixer = nn.Sequential(*[ResidualBlock(cfg.hidden_dim) for _ in range(cfg.mixer_layers)])
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.output_projection = nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.vocab_size)

        def forward(self, tokens):
            if tokens.ndim != 3:
                raise ValueError("tokens must have shape (batch, context_frames, positions)")
            if tokens.shape[1] != self.config.context_frames:
                raise ValueError("context_frames dimension does not match config")
            if tokens.shape[2] != self.config.positions:
                raise ValueError("positions dimension does not match config")
            if tokens.dtype != torch.long:
                tokens = tokens.long()

            embedded = self.token_embedding(tokens)
            embedded = embedded + self.position_embedding.view(1, 1, self.config.positions, self.config.embed_dim)
            embedded = embedded + self.frame_embedding.view(1, self.config.context_frames, 1, self.config.embed_dim)
            frame_summaries = embedded.mean(dim=2)
            hidden = self.frame_projection(frame_summaries)
            hidden = self.mixer(hidden)
            context_state = self.output_norm(hidden[:, -1, :])
            repeated_state = context_state.unsqueeze(1).expand(-1, self.config.positions, -1)
            repeated_pos = self.position_embedding.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            return self.output_projection(torch.cat([repeated_state, repeated_pos], dim=-1)).float()

    return _TinyFramePredictor(config)


def _config_from_profile(
    config_or_profile: TinyFramePredictorConfig | TinyFramePredictorProfileConfig | str,
    *,
    context_frames: int | None = None,
    vocab_size: int | None = None,
) -> tuple[str | None, TinyFramePredictorConfig]:
    if isinstance(config_or_profile, str):
        profile_config = load_tiny_frame_predictor_profile(config_or_profile)
        config = TinyFramePredictorConfig(
            context_frames=profile_config.context_frames if context_frames is None else int(context_frames),
            positions=profile_config.positions,
            vocab_size=profile_config.vocab_size if vocab_size is None else int(vocab_size),
            embed_dim=profile_config.embed_dim,
            hidden_dim=profile_config.hidden_dim,
            mixer_layers=profile_config.mixer_layers,
        )
        return profile_config.profile, config
    if isinstance(config_or_profile, TinyFramePredictorProfileConfig):
        config = TinyFramePredictorConfig(
            context_frames=config_or_profile.context_frames if context_frames is None else int(context_frames),
            positions=config_or_profile.positions,
            vocab_size=config_or_profile.vocab_size if vocab_size is None else int(vocab_size),
            embed_dim=config_or_profile.embed_dim,
            hidden_dim=config_or_profile.hidden_dim,
            mixer_layers=config_or_profile.mixer_layers,
        )
        return config_or_profile.profile, config
    config = TinyFramePredictorConfig(
        context_frames=config_or_profile.context_frames if context_frames is None else int(context_frames),
        positions=config_or_profile.positions,
        vocab_size=config_or_profile.vocab_size if vocab_size is None else int(vocab_size),
        embed_dim=config_or_profile.embed_dim,
        hidden_dim=config_or_profile.hidden_dim,
        mixer_layers=config_or_profile.mixer_layers,
    )
    return None, config


def _resolve_runtime_device(torch, device: str) -> tuple[str, str | None]:
    normalized = device.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda", None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", None
        return "cpu", None
    if normalized == "cuda" and torch.cuda.is_available():
        return "cuda", None
    if normalized == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", None
    if normalized == "cpu":
        return "cpu", None
    return "cpu", f"requested {normalized} runtime is unavailable; using cpu"


def _resolve_runtime_dtype(torch, dtype: str, *, device: str) -> tuple[object, str, str | None]:
    normalized = dtype.strip().lower()
    if normalized not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    if normalized == "auto":
        return torch.float32, "float32", None
    if normalized == "float32":
        return torch.float32, "float32", None
    if device == "cuda":
        if normalized == "float16":
            return torch.float16, "float16", None
        return torch.bfloat16, "bfloat16", None
    if device == "mps" and normalized == "float16":
        return torch.float16, "float16", None
    return torch.float32, "float32", f"requested {normalized} dtype is unsupported on {device}; using float32"


def _initialize_deterministic_tiny_frame_predictor(model, *, torch) -> None:
    phase = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 1:
                if name.endswith("norm.weight"):
                    param.fill_(1.0)
                else:
                    param.zero_()
                continue
            values = torch.arange(param.numel(), dtype=torch.float32, device=param.device).reshape(param.shape)
            scale = 0.02 / max(float(param.shape[-1]), 1.0)
            param.copy_((torch.sin(values + phase) * scale).to(dtype=param.dtype))
            phase += float(param.numel())


def load_tiny_frame_predictor_runtime(
    profile: str | TinyFramePredictorProfileConfig | TinyFramePredictorConfig,
    *,
    context_frames: int | None = None,
    vocab_size: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> TinyFramePredictor:
    torch, _ = _require_torch()
    profile_name, config = _config_from_profile(profile, context_frames=context_frames, vocab_size=vocab_size)
    resolved_device, device_fallback = _resolve_runtime_device(torch, device)
    torch_dtype, resolved_dtype, dtype_fallback = _resolve_runtime_dtype(torch, dtype, device=resolved_device)
    fallback_parts = [part for part in (device_fallback, dtype_fallback) if part]

    model = build_tiny_frame_predictor(config)
    model = model.to(device=resolved_device, dtype=torch_dtype)
    model.eval()
    _initialize_deterministic_tiny_frame_predictor(model, torch=torch)

    return TinyFramePredictor(
        model,
        config=config,
        profile=profile_name or "custom",
        torch=torch,
        device=resolved_device,
        dtype=resolved_dtype,
        fallback_reason="; ".join(fallback_parts) if fallback_parts else None,
    )


def summarize_tiny_frame_predictor(config_or_profile: TinyFramePredictorConfig | TinyFramePredictorProfileConfig | str):
    profile_name, config = _config_from_profile(config_or_profile)

    model = build_tiny_frame_predictor(config)
    parameter_count = sum(int(param.numel()) for param in model.parameters())
    payload = {
        "command": "lossless_tiny_frame_predictor_summary",
        "profile": profile_name,
        "parameter_count": parameter_count,
    }
    payload.update(asdict(config))
    return payload
