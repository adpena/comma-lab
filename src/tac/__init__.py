"""task-aware-codec (tac) — Learned post-filters for task-aware video compression.

Train tiny CNN post-filters that correct decoded video frames by backpropagating
through frozen perception networks (scorers). The filter learns corrections that
minimize the scorer's distortion metric, not generic pixel quality.

Modules:
  - tac.architectures: 7 architectures (PostFilter, Dilated, PixelShuffle, PSD,
    Depthwise, Luma, FiLM) + 12 variant aliases
  - tac.training: Trainer (QAT+EMA+SWA, best-checkpoint, lazy loading, resume),
    EMA, SWA, KalmanWeightFilter
  - tac.losses: scorer_loss (train), eval_scorer_loss (hard argmax), segnet_ste_loss, saliency recon
  - tac.data: video decoding, lazy pair construction, saliency loading
  - tac.quantization: FakeQuant STE, LSQ, QATPostFilter, int8 save/load
  - tac.scorer: scoring formula, sensitivity analysis, load_scorers, detect_device
  - tac.evaluate: proxy scoring, top-K checkpoint averaging, checkpoint discovery

Quick start::

    from tac.architectures import build_postfilter
    from tac.training import Trainer, TrainConfig
    from tac.data import decode_archive, decode_video, build_pairs

    model = build_postfilter("standard", hidden=64)
    config = TrainConfig(hidden=64, epochs=1000, alpha=20, tag="my_run")
    trainer = Trainer(model, config, device="mps")
    trainer.fit(comp_pairs, gt_pairs, posenet, segnet, sal_weights)
"""

__version__ = "0.7.0"
