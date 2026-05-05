"""Score-band predictor with refusal modes.

Council 22/0 verdict 2026-05-05 (`feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`):
the predictor must (a) model distortion, (b) refuse to emit when calibration
support is insufficient, (c) sanity-gate against monotonic-quality reasoning
(lossy-better-than-lossless is incoherent).

This module replaces the rate-only `predicted_score_band` lookup that drove the
apogee_int4 8x miss (predicted [0.155, 0.180]; landed 1.4287 [contest-CUDA]).
"""

from tac.predictor.score_band import (
    CalibrationAnchor,
    DistortionProxy,
    ScoreBand,
    fit_distortion_curve,
    load_calibration_anchors,
    predict_score_band,
)

__all__ = [
    "CalibrationAnchor",
    "DistortionProxy",
    "ScoreBand",
    "fit_distortion_curve",
    "load_calibration_anchors",
    "predict_score_band",
]
