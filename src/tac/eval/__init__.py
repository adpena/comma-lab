"""Authoritative evaluation for the comma video compression challenge.

This package provides the canonical, platform-agnostic evaluation pipeline.
All platform-specific code (Modal, Lightning, Kaggle) imports from here.

Usage::

    from tac.eval import AuthEvaluator

    evaluator = AuthEvaluator(upstream_dir=Path("/path/to/upstream"))
    results = evaluator.eval_checkpoint(
        checkpoint_path=Path("renderer_best.bin"),
        archive_size_bytes=204_800,
    )
    print(results.score)
"""

from .auth_eval import (
    EXPECTED_FRAME_BYTES,
    EXPECTED_RAW_BYTES,
    FALLBACK_UNCOMPRESSED_SIZE,
    NUM_FRAMES,
    OUT_H,
    OUT_W,
    SEG_H,
    SEG_W,
    AuthEvaluator,
    AuthResult,
    RendererMode,
    ReportMetrics,
    compute_final_score,
    parse_report,
    parse_report_file,
    run_evaluate_py,
    score_breakdown,
    validate_raw_file,
)

__all__ = [
    # Constants
    "EXPECTED_FRAME_BYTES",
    "EXPECTED_RAW_BYTES",
    "FALLBACK_UNCOMPRESSED_SIZE",
    "NUM_FRAMES",
    "OUT_H",
    "OUT_W",
    "SEG_H",
    "SEG_W",
    # Classes
    "AuthEvaluator",
    "AuthResult",
    "RendererMode",
    "ReportMetrics",
    # Functions — report parsing & scoring
    "compute_final_score",
    "parse_report",
    "parse_report_file",
    "run_evaluate_py",
    "score_breakdown",
    "validate_raw_file",
]
