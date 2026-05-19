# SPDX-License-Identifier: MIT
"""Task-Aware Compression (tac) - scorer-aware compression primitives.

`tac` is the reusable algorithmic layer for compression optimized against
downstream machine-perception tasks. It includes codecs, renderers, packet
compilers, quantizers, scorer contracts, sensitivity maps, and optimization
helpers. In this codebase, "codec" names a concrete encoder/decoder or wire
format; "Task-Aware Compression" names the broader package and research stack.

Modules:
  - tac.architectures: 8 architectures (PostFilter, Dilated, PixelShuffle, PSD,
    Depthwise, Luma, FiLM, PairAware) + 12 variant aliases
  - tac.training: Trainer (QAT+EMA+SWA, best-checkpoint, lazy loading, resume),
    EMA, SWA, KalmanWeightFilter
  - tac.losses: scorer_loss (train), eval_scorer_loss (hard argmax), segnet_ste_loss, saliency recon
  - tac.data: video decoding, lazy pair construction, saliency loading
  - tac.quantization: FakeQuant STE, LSQ, QATPostFilter, int8 save/load
  - tac.scorer: scoring formula, sensitivity analysis, load_scorers, detect_device
  - tac.evaluate: proxy scoring, top-K checkpoint averaging, checkpoint discovery
  - tac.models: Pydantic models (ScoreResult, CheckpointMeta, TrainConfig)

Quick start::

    from tac import Trainer, TrainConfig, build_postfilter
    model = build_postfilter("standard", hidden=64)
    config = TrainConfig(hidden=64, epochs=1000, alpha=20, tag="my_run")
    trainer = Trainer(model, config, device="mps")
    trainer.fit(comp_pairs, gt_pairs, posenet, segnet, sal_weights)
"""

from __future__ import annotations

__version__ = "1.0.5"

# ── Lazy public API ──────────────────────────────────────────────────────
# Heavy imports (torch, pydantic) are deferred so that `import tac` stays
# fast for CLI tooling and introspection.
_LAZY_PUBLIC_API = {
    # tac.training
    "Trainer": (".training", "Trainer"),
    "TrainConfig": (".training", "TrainConfig"),
    # tac.architectures
    "build_postfilter": (".architectures", "build_postfilter"),
    # tac.renderer
    "build_renderer": (".renderer", "build_renderer"),
    # tac.models
    "ScoreResult": (".models", "ScoreResult"),
    "CheckpointMeta": (".models", "CheckpointMeta"),
    "AveragedCheckpoint": (".models", "AveragedCheckpoint"),
    "SensitivityResult": (".models", "SensitivityResult"),
}


def __getattr__(name: str):
    """Lazy-load public API symbols on first access."""
    if name in _LAZY_PUBLIC_API:
        module_path, attr = _LAZY_PUBLIC_API[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy public symbols to interactive help without importing them."""
    return sorted(set(globals()) | set(_LAZY_PUBLIC_API))


__all__ = [
    "AveragedCheckpoint",
    "CheckpointMeta",
    "ScoreResult",
    "SensitivityResult",
    "TrainConfig",
    "Trainer",
    "__version__",
    "build_postfilter",
    "build_renderer",
]
