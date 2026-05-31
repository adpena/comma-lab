# Codex findings: boundary-modeling SegNet resolution hardening

`[macOS-MLX research-signal]` / design-review memo. No score claim, no promotion
authority, no dispatch authority.

## Findings

1. The boundary-modeling memo over-reduced the SegNet distortion grid. Current
   upstream code resizes the last frame to `(384, 512)` in
   `SegNet.preprocess_input`, and `compute_distortion` averages argmax
   disagreement over the SegNet output. A live dummy forward returned
   `preprocess_shape=(1,3,384,512)` and `output_shape=(1,5,384,512)`.
2. EfficientNet-B2 internal stride/downsampling is real feature-path context, but
   it is not the direct distortion-grid resolution. It can motivate a
   frequency-response probe; it cannot by itself justify discarding sub-256px or
   other high-frequency boundary signal.
3. The corrected operator bridge should compare contest video against candidate
   inflated video through SegNet argmax maps, margin/boundary maps, class-pair
   transition surfaces, semantic regions, and pair-level PoseNet/SegNet tails
   before selecting MLX loss modes or frequency gates.
4. The mathematically score-faithful SegNet repair objective is not a soft
   probability-matching loss. KL, target-vs-rest TCKD, and guarded decision-KD
   can still reward teacher softness at exactly the boundary cells where `d_seg`
   needs a hard argmax decision. The Crammer-Singer all-impostor hinge on raw
   logits targets `argmax(candidate)==argmax(source)` with a configurable margin
   buffer and is the correct local surrogate for SegNet `d_seg` repair.

## Patch

Updated `.omx/research/boundary_modeling_derived_vs_learned_multiscale_20260531.md`
to:

- replace the `<=256px`/stride-2-as-distortion-grid claim with the verified
  384x512 scorer grid;
- preserve EfficientNet stride/downsampling as feature-extraction and
  receptive-field context only;
- require empirical frequency-response / argmax-visual evidence before dropping
  high-frequency boundary signal;
- route the bridge through existing tooling:
  `tools/xray_pair_component_errors.py`,
  `tools/build_segnet_boundary_marginals.py`,
  `src/tac/analysis/segnet_boundary_marginals.py`,
  `src/tac/xray/segnet_margin_polytope.py`,
  `src/tac/visualization/segnet_viz.py`,
  `src/tac/visualization/comparison_video.py`, and
  `src/tac/research/generate_visual_comparison.py`;
- keep the output explicitly non-authoritative:
  `[macOS-MLX research-signal]`, `score_claim=false`, no rank/kill/promote
  authority, and no replacement for `[contest-CPU]` / `[contest-CUDA]`.
- update loss-routing guidance so `boundary_argmax_hinge_loss` is the
  `d_seg`-faithful training objective, while KL/TCKD/decision-KD remain
  diagnostic arms or ablation controls.

## Remaining blockers

- No candidate-side bridge artifact has been run yet.
- No empirical wavelet/frequency-response probe has established which bands alter
  the 384x512 argmax or margin surface.
- The optimal hinge margin is not known yet; it should be learned/derived from
  full-dataset margin gaps, semantic region, runtime perturbation, postfilter
  noise, and exact-axis drift.
