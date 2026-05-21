# LL Frame/Pair Curriculum Pair-4 Guarded Artifact Landed

**Author**: codex  
**UTC**: 2026-05-21T00:48:53Z  
**Tool**: `tools/build_ll_frame_pair_curriculum.py`

## Artifact

Generated:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.md`

Inputs:

- `master_gradient_frame_axis_l1_20260520_codex.npy`
- `master_gradient_frame_decomposition_20260520_codex.json`
- `ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json`

## Result

The current LL curriculum is now built from both:

1. the per-frame/per-pair master-gradient topology, and
2. the guarded LL response plan that carries the pair #4 seed-wrap prohibition.

Summary:

- topology: `non_overlapping`
- frames: `16`
- pairs: `8`
- adjustment layers: `11`
- score claim: `false`
- knob bias: `prefer_byte_neutral_masked_runtime_knobs_before_charged_sparse_pixels`

The response policy carries both prohibitions:

- `do_not_widen_coordinate_sparse_residual_sidecar`
- `do_not_wrap_procedural_seed_bytes_with_magic_codec`

## Top routing targets

Top frames:

`15, 13, 5, 9, 11, 7, 1, 3`

All top-eight frames are odd-indexed pair-last frames, matching the scorer
topology: SegNet sees the last frame of each non-overlapping pair while PoseNet
sees both frames.

Top pairs:

`7, 6, 4, 2, 5, 3, 1, 0`

The first two pairs carry both SegNet and PoseNet adjustment layers:

- pair `7`, frames `(14, 15)`: seg share `0.636`, pose share `0.364`
- pair `6`, frames `(12, 13)`: seg share `0.670`, pose share `0.330`

## Interpretation

This is a planning/curriculum artifact, not score evidence. It says the next
LL/surrogate or byte-neutral correction work should prioritize pair-last
SegNet-visible frames, especially pairs `7` and `6`, while keeping procedural
seed bytes raw and refusing charged sparse residual widening until a residual
grammar can clear its break-even gate.

## Verification

Command run:

```bash
.venv/bin/python tools/build_ll_frame_pair_curriculum.py --frame-axis-npy experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_axis_l1_20260520_codex.npy --decomposition-json experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_decomposition_20260520_codex.json --response-plan-json experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json --json-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.json --md-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.md
```

Output:

- `n_frames`: `16`
- `n_pairs`: `8`
- `adjustment_layers`: `11`
- `score_claim`: `false`

