# Codex Findings - Z8 Pixel Driver And SegNet Grid Premise - 2026-05-31T15:30:38Z

## Scope

Adversarial review of the current predictive hierarchical coding stack work
after the Z8 runtime payload bridge landing, with one narrow `.omx` premise
audit from subagent `019e7ea2-1d85-7232-a843-5602a1860be4`.

## Finding 1 - Z8 WZ/Mamba State Is Now A Receiver-Side Pixel Driver

The Z8 receiver path now consumes decoded Wyner-Ziv/Mamba top states by
projecting them into the frame-1 Mallat top-LL image space before inverse
wavelet reconstruction. This is intentionally still false authority for score
or exact promotion: the receiver consumes the payload, but exact-axis score
authority still requires archive-bound receiver proof, valid semantic
byte-mutation proof, lane preclaim, and CPU/CUDA adjudication.

Current code surfaces:

- `src/tac/substrates/z8_hierarchical_predictive_coding/runtime_payload_bridge.py`
  decodes WZ top states and applies a bounded deterministic top-LL projection.
- `src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py` applies
  that projection in the generated receiver runtime.
- `src/tac/substrates/z8_hierarchical_predictive_coding/archive_candidate.py`
  vendors the bridge file and labels the transform as a WZ/Mamba top-LL pixel
  driver while keeping exact authority false.

## Finding 2 - Supersede The "SegNet Interiors Are Free" Premise

The memo
`.omx/research/z8_yousfi_suniward_cost_map_training_weight_design_20260531.md`
contains a stale optimization premise around lines 20-25: it says SegNet
responds only to class-boundary argmax flips, that the stride-2 stem is blind
below `(256, 192)`, and that class interiors are effectively free.

That premise must not drive default acquisition or training weights. It is
superseded by
`.omx/research/boundary_modeling_derived_vs_learned_multiscale_20260531.md`,
which correctly keeps the scorer grounded in the 384x512 SegNet argmax output
grid. Internal stride can inform priors, but it does not license discarding
higher-frequency boundary, class-interior, or semantic-region evidence.

Operational consequence:

- Treat inverse S-UNIWARD / texture-undetectability weighting as an ablation,
  not the default.
- Default score-lowering repair/training weights should come from measured
  SegNet/PoseNet response surfaces at class, boundary, region, frame, and
  full-video scopes.
- The missing reusable surface is a shared MLX `recon_pixel_weight` path that
  can consume a 384x512 boundary/margin-derived map and run A/B against
  inverse S-UNIWARD texture weights without conflating the premises.

## Verification

- `ruff` passed on the touched Z8 bridge/runtime/contract files.
- `pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_archive_candidate_bridge.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_inflate_mallat_wavelet_archive_consumption.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_basic.py -q`
  passed with 43 tests.
- `pytest src/tac/tests/test_archive_bound_candidate_adapter_spine.py -q`
  passed with 11 tests.
- `tools/review_tracker.py policy-check` passed after two reviewed marks on
  the new bridge file.

## Next Concrete Slice

Add the shared MLX `recon_pixel_weight` channel at
`src/tac/substrates/_shared/mlx_score_aware/bundle.py` and
`src/tac/substrates/_shared/mlx_score_aware/loss.py`, then feed Z8 a
384x512 SegNet boundary/margin map as the default A/B against inverse S-UNIWARD
texture weighting.
