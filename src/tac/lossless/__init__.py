# SPDX-License-Identifier: MIT
"""Lossless compression subsystem for commavq-style experiments."""

from .arithmetic import (
    FRAME_BOS_TOKEN,
    SEGMENT_EOT_TOKEN,
    GPTArithmeticEstimate,
    GPTArithmeticPlan,
    GPTArithmeticProfileConfig,
    build_gpt_arithmetic_plan,
    estimate_gpt_arithmetic_workload,
    flatten_tokens_for_gpt_arithmetic,
    load_gpt_arithmetic_profile,
    materialize_gpt_arithmetic_stream,
)
from .codecs import (
    build_lzma_baseline_submission,
    build_zpaq_baseline_submission,
    compress_lossless_file,
    decompress_lossless_file,
    evaluate_lzma_baseline_submission,
    evaluate_zpaq_baseline_submission,
    lzma_roundtrip_file,
    zpaq_roundtrip_file,
)
from .contracts import LosslessCompressionResult, LosslessVerificationResult
from .data import token_byte_length, token_bytes
from .evaluate import (
    compression_rate,
    evaluate_commavq_dataset_archive,
    evaluate_lossless_archive,
    verify_exact_tokens,
)
from .frequency_coder import FrequencyEncodedStream, decode_uint16_frequency_stream, encode_uint16_frequency_stream
from .profiles import PROFILES
from .state import load_lossless_result, promote_lossless_result, render_lossless_latest
from .submission import build_submission_zip, validate_submission_inputs


def __getattr__(name: str):
    """Lazy import for heavy gpt_score module to avoid importing torch at package load time."""
    if name == "score_commavq_gpt_sample":
        from .gpt_score import score_commavq_gpt_sample
        return score_commavq_gpt_sample
    if name == "score_tokens_with_logits_fn":
        from .gpt_score import score_tokens_with_logits_fn
        return score_tokens_with_logits_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FRAME_BOS_TOKEN",
    "PROFILES",
    "SEGMENT_EOT_TOKEN",
    "FrequencyEncodedStream",
    "GPTArithmeticEstimate",
    "GPTArithmeticPlan",
    "GPTArithmeticProfileConfig",
    "LosslessCompressionResult",
    "LosslessVerificationResult",
    "build_gpt_arithmetic_plan",
    "build_lzma_baseline_submission",
    "build_submission_zip",
    "build_zpaq_baseline_submission",
    "compress_lossless_file",
    "compression_rate",
    "decode_uint16_frequency_stream",
    "decompress_lossless_file",
    "encode_uint16_frequency_stream",
    "estimate_gpt_arithmetic_workload",
    "evaluate_commavq_dataset_archive",
    "evaluate_lossless_archive",
    "evaluate_lzma_baseline_submission",
    "evaluate_zpaq_baseline_submission",
    "flatten_tokens_for_gpt_arithmetic",
    "load_gpt_arithmetic_profile",
    "load_lossless_result",
    "lzma_roundtrip_file",
    "materialize_gpt_arithmetic_stream",
    "promote_lossless_result",
    "render_lossless_latest",
    "score_commavq_gpt_sample",
    "score_tokens_with_logits_fn",
    "token_byte_length",
    "token_bytes",
    "validate_submission_inputs",
    "verify_exact_tokens",
    "zpaq_roundtrip_file",
]
