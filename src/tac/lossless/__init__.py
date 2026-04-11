"""Lossless compression subsystem for commavq-style experiments."""

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
from .arithmetic import (
    FRAME_BOS_TOKEN,
    SEGMENT_EOT_TOKEN,
    GPTArithmeticEstimate,
    GPTArithmeticPlan,
    GPTArithmeticProfileConfig,
    build_gpt_arithmetic_plan,
    estimate_gpt_arithmetic_workload,
    flatten_tokens_for_gpt_arithmetic,
    materialize_gpt_arithmetic_stream,
    load_gpt_arithmetic_profile,
)
from .contracts import LosslessCompressionResult, LosslessVerificationResult
from .data import token_byte_length, token_bytes
from .evaluate import (
    compression_rate,
    evaluate_commavq_dataset_archive,
    evaluate_lossless_archive,
    verify_exact_tokens,
)
from .profiles import PROFILES
from .state import load_lossless_result, promote_lossless_result, render_lossless_latest
from .submission import build_submission_zip, validate_submission_inputs

__all__ = [
    "LosslessCompressionResult",
    "LosslessVerificationResult",
    "FRAME_BOS_TOKEN",
    "SEGMENT_EOT_TOKEN",
    "GPTArithmeticEstimate",
    "GPTArithmeticPlan",
    "GPTArithmeticProfileConfig",
    "PROFILES",
    "build_lzma_baseline_submission",
    "build_zpaq_baseline_submission",
    "build_gpt_arithmetic_plan",
    "estimate_gpt_arithmetic_workload",
    "compress_lossless_file",
    "compression_rate",
    "decompress_lossless_file",
    "evaluate_lzma_baseline_submission",
    "evaluate_zpaq_baseline_submission",
    "evaluate_commavq_dataset_archive",
    "evaluate_lossless_archive",
    "flatten_tokens_for_gpt_arithmetic",
    "materialize_gpt_arithmetic_stream",
    "build_submission_zip",
    "load_lossless_result",
    "load_gpt_arithmetic_profile",
    "promote_lossless_result",
    "render_lossless_latest",
    "token_byte_length",
    "token_bytes",
    "lzma_roundtrip_file",
    "zpaq_roundtrip_file",
    "validate_submission_inputs",
    "verify_exact_tokens",
]
