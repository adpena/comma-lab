"""Named lossless experiment profiles for commavq-style work."""

from __future__ import annotations

from dataclasses import dataclass

LZMA_BASELINE = {
    "method": "lzma",
    "level": 6,
}

ZPAQ_BASELINE = {
    "method": "zpaq",
    "preset": "max",
}

GPT_ARITHMETIC_SMALL = {
    "method": "gpt_arithmetic",
    "model": "small",
    "context_tokens": 256,
}

GPT_NEXT_FRAME_SMALL = {
    "method": "gpt_next_frame",
    "model": "small",
    "context_frames": 1,
}

GPT_ARITHMETIC_LARGE = {
    "method": "gpt_arithmetic",
    "model": "large",
    "context_tokens": 1024,
}

TINY_FRAME_PREDICTOR_SMALL = {
    "method": "tiny_frame_predictor",
    "context_frames": 8,
    "positions": 128,
    "vocab_size": 1025,
    "embed_dim": 64,
    "hidden_dim": 128,
    "mixer_layers": 2,
}

PREV_SYMBOL_POSITION_MAJOR = {
    "method": "prev_symbol_position_major",
}

GLOBAL_PREV_SYMBOL_POSITION_MAJOR = {
    "method": "global_prev_symbol_position_major",
    "chunk_count": 1,
}

NEURAL_CODEC_SMOKE = {
    "method": "self_compressing_nn",
    "epochs": 1,
    "smoke": True,
}

PROFILES = {
    "lzma_baseline": LZMA_BASELINE,
    "zpaq_baseline": ZPAQ_BASELINE,
    "prev_symbol_position_major": PREV_SYMBOL_POSITION_MAJOR,
    "global_prev_symbol_position_major": GLOBAL_PREV_SYMBOL_POSITION_MAJOR,
    "gpt_arithmetic_small": GPT_ARITHMETIC_SMALL,
    "gpt_next_frame_small": GPT_NEXT_FRAME_SMALL,
    "gpt_arithmetic_large": GPT_ARITHMETIC_LARGE,
    "tiny_frame_predictor_small": TINY_FRAME_PREDICTOR_SMALL,
    "neural_codec_smoke": NEURAL_CODEC_SMOKE,
}


@dataclass(frozen=True)
class GPTNextFrameProfileConfig:
    profile: str
    method: str
    model: str
    context_frames: int

    def __post_init__(self) -> None:
        if self.method != "gpt_next_frame":
            raise ValueError(f"unsupported next-frame method: {self.method}")
        if self.model != "small":
            raise ValueError(f"unsupported next-frame model: {self.model}")
        if self.context_frames <= 0:
            raise ValueError("context_frames must be positive")


def load_gpt_next_frame_profile(profile: str) -> GPTNextFrameProfileConfig:
    try:
        config = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown gpt next-frame profile: {profile}") from exc

    return GPTNextFrameProfileConfig(
        profile=profile,
        method=str(config.get("method", "")),
        model=str(config.get("model", "")),
        context_frames=int(config.get("context_frames", 0)),
    )


@dataclass(frozen=True)
class TinyFramePredictorProfileConfig:
    profile: str
    method: str
    context_frames: int
    positions: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    mixer_layers: int

    def __post_init__(self) -> None:
        if self.method != "tiny_frame_predictor":
            raise ValueError(f"unsupported tiny frame predictor method: {self.method}")
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


def load_tiny_frame_predictor_profile(profile: str) -> TinyFramePredictorProfileConfig:
    try:
        config = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown tiny frame predictor profile: {profile}") from exc

    return TinyFramePredictorProfileConfig(
        profile=profile,
        method=str(config.get("method", "")),
        context_frames=int(config.get("context_frames", 0)),
        positions=int(config.get("positions", 0)),
        vocab_size=int(config.get("vocab_size", 0)),
        embed_dim=int(config.get("embed_dim", 0)),
        hidden_dim=int(config.get("hidden_dim", 0)),
        mixer_layers=int(config.get("mixer_layers", 0)),
    )
