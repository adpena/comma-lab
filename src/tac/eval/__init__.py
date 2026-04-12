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

from .auth_eval import AuthEvaluator, RendererMode

__all__ = ["AuthEvaluator", "RendererMode"]
