"""task-aware-codec (tac) — Learned post-filters for task-aware video compression.

Train tiny CNN post-filters that correct decoded video frames by backpropagating
through frozen perception networks (scorers). The filter learns corrections that
minimize the scorer's distortion metric, not generic pixel quality.

Modules:
  - tac.architectures: PostFilter, DilatedPostFilter, PixelShufflePostFilter, PSD
  - tac.training: Trainer class (QAT+EMA, best-checkpoint int8 selection)
  - tac.losses: scorer_loss, segnet_ste_loss, boundary weighting
  - tac.data: video decoding, pair construction, saliency loading
  - tac.quantization: int8 save/load, FakeQuant STE, LSQ
  - tac.scorer: scoring formula, sensitivity analysis

Quick start::

    from tac.architectures import build_postfilter
    from tac.training import Trainer, TrainConfig
    from tac.data import decode_archive, decode_video, build_pairs

    model = build_postfilter("standard", hidden=64)
    config = TrainConfig(hidden=64, epochs=1000, alpha=20, tag="my_run")
    trainer = Trainer(model, config, device="mps")
    trainer.fit(comp_pairs, gt_pairs, posenet, segnet, sal_weights)
"""

__version__ = "0.4.0"
