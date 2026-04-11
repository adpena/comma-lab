"""Named lossless experiment profiles for commavq-style work."""

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

GPT_ARITHMETIC_LARGE = {
    "method": "gpt_arithmetic",
    "model": "large",
    "context_tokens": 1024,
}

PREV_SYMBOL_POSITION_MAJOR = {
    "method": "prev_symbol_position_major",
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
    "gpt_arithmetic_small": GPT_ARITHMETIC_SMALL,
    "gpt_arithmetic_large": GPT_ARITHMETIC_LARGE,
    "neural_codec_smoke": NEURAL_CODEC_SMOKE,
}
