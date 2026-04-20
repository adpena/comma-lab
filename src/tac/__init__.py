"""task-aware-codec (tac) — Learned post-filters for task-aware video compression.

Train tiny CNN post-filters that correct decoded video frames by backpropagating
through frozen perception networks (scorers). The filter learns corrections that
minimize the scorer's distortion metric, not generic pixel quality.

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


def __getattr__(name: str):
    """Lazy-load public API symbols on first access."""
    _API_MAP = {
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
    if name in _API_MAP:
        module_path, attr = _API_MAP[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # Training
    "Trainer",
    "TrainConfig",
    # Architectures
    "build_postfilter",
    # Renderer
    "build_renderer",
    # Data models
    "ScoreResult",
    "CheckpointMeta",
    "AveragedCheckpoint",
    "SensitivityResult",
]
