# Wave A-1 Real-Scorer Regression Kit (2026-05-12)

## Scope

Completed the partner WAVE-A-1 scorer-preprocess hardening work by adding the
missing shared regression kit used by the 15 substrate sister tests.

## Bug Class

WWW4/FIX-H showed a scorer-contract failure mode: score-aware losses can pass
5D `(B, T, C, H, W)` tensors directly into `SegNet.forward()`, which expects
4D post-`preprocess_input` tensors. Dummy smoke scorers accepted the wrong shape
and missed the crash.

## Patch

Added `tac.substrates._shared.score_aware_loss_real_scorer_test_kit`:

- real `upstream.modules.SegNet` with random weights, exercising the true
  SMP/Unet 4D input contract;
- upstream-contract PoseNet stand-in to keep tests CPU-fast while preserving
  `preprocess_input` shape behavior;
- random-frame and PyAV-decoded contest-video helpers;
- optional `extra_kwargs_factory` hook for substrates with commitment,
  wavelet, hyperprior, or residual auxiliary loss terms.

## Evidence

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/*/tests/test_score_aware_loss_real_scorer_forward.py -q`

Result: 34 passed in 108.50s. The real scorer stack and contest video are
available in this environment, so all sister tests exercise the SegNet contract
and PyAV-decoded real-frame path.

## Score-Lowering Relevance

This clears the scorer-preprocess bug class that blocked substrate
first-anchors. SIREN, Ballé, VQ-VAE, Cool-Chic, wavelet, PR101 clone, and the
NeRV-family sister substrates can now share one regression harness before any
Vast/Modal dispatch.
