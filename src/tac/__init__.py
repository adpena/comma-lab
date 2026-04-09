"""task-aware-codec (tac) — Learned post-filters for task-aware video compression.

Train tiny CNN post-filters that correct decoded video frames by backpropagating
through frozen perception networks (scorers). The filter learns corrections that
minimize the scorer's distortion metric, not generic pixel quality.

Key components:
  - tac.architectures: PostFilter, DilatedPostFilter, PixelShufflePostFilter, PSD
  - tac.training: QAT+EMA training loop with saliency weighting
  - tac.quantization: int8 save/load with per-channel and LSQ support
  - tac.evaluation: faithful proxy scorer, automatic triage
  - tac.scheduler: multi-platform experiment orchestration

Quick start::

    from tac import PostFilter, Trainer, Scorer

    scorer = Scorer.from_safetensors("posenet.safetensors", "segnet.safetensors")
    model = PostFilter(hidden=64, variant="psd")
    trainer = Trainer(model, scorer, alpha=20, epochs=1000)
    trainer.fit("archive.zip", "ground_truth.mkv")
    trainer.save_best_int8("filter.pt")
"""

__version__ = "0.1.0"
