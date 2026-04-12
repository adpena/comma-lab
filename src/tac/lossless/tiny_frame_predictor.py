from __future__ import annotations

from dataclasses import asdict, dataclass

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
    pass


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


def summarize_tiny_frame_predictor(config_or_profile: TinyFramePredictorConfig | TinyFramePredictorProfileConfig | str):
    if isinstance(config_or_profile, str):
        profile_config = load_tiny_frame_predictor_profile(config_or_profile)
        config = TinyFramePredictorConfig(
            context_frames=profile_config.context_frames,
            positions=profile_config.positions,
            vocab_size=profile_config.vocab_size,
            embed_dim=profile_config.embed_dim,
            hidden_dim=profile_config.hidden_dim,
            mixer_layers=profile_config.mixer_layers,
        )
        profile_name = profile_config.profile
    elif isinstance(config_or_profile, TinyFramePredictorProfileConfig):
        config = TinyFramePredictorConfig(
            context_frames=config_or_profile.context_frames,
            positions=config_or_profile.positions,
            vocab_size=config_or_profile.vocab_size,
            embed_dim=config_or_profile.embed_dim,
            hidden_dim=config_or_profile.hidden_dim,
            mixer_layers=config_or_profile.mixer_layers,
        )
        profile_name = config_or_profile.profile
    else:
        config = config_or_profile
        profile_name = None

    model = build_tiny_frame_predictor(config)
    parameter_count = sum(int(param.numel()) for param in model.parameters())
    payload = {
        "command": "lossless_tiny_frame_predictor_summary",
        "profile": profile_name,
        "parameter_count": parameter_count,
    }
    payload.update(asdict(config))
    return payload

