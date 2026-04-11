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
