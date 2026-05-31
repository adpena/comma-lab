# Codex Findings: Mamba/Dreamer/HPC Variational Implications

timestamp_utc: 2026-05-31T17:44:46Z
agent: codex
lane_id: lane_codex_mamba_dreamer_hpc_variational_20260531
scope: read-only research/explorer follow-up; no source edits
authority: findings memo only; not a score claim; not promotion authority

## Executive verdict

Z8 joint P18/P19 is the right place to attach a serious Mamba/Dreamer/HPC rate
attack, but only if it becomes a deterministic full-video direct-transform
codec:

1. direct transform of pixels or residuals;
2. score-aware full-video adjoint/VJP into transform coefficients;
3. quantization, deadzone, and mode allocation in coefficient space;
4. real entropy coding of the charged residual/mode stream;
5. exact archive/runtime inflation and exact CPU/CUDA evaluation.

Mamba and Dreamer are useful as temporal/context priors and allocators. They are
not authority by themselves. Hidden state, MLX-local state, advisory loss curves,
or decoded tensor parity remain false authority unless their bytes are consumed
by `inflate.sh archive_dir output_dir file_list` and replay through the contest
scorer axis.

## Current quantitative constraint

Current Z8 600-pair advisory artifact is rate-bound but not only rate-bound:

- byte-closed archive zip: 152,200,539 bytes;
- advisory score: 104.93672293420708;
- rate term: 101.34409156317021;
- SegNet term: 1.2953101247549057;
- PoseNet term: 2.297321246281969;
- combined distortion terms: 3.5926313710368746.

Therefore byte reduction alone cannot reach the current CPU frontier. Even a
zero-byte Z8 packet with the same decoded frames would still score above 3.59.
To reach the current CPU frontier score 0.1919853363, the codec needs both:

- archive size in the approximate 150k-300k byte regime, depending distortion;
- frontier-class SegNet/PoseNet distortions, not the current Z8 advisory
  distortions.

Rate-only byte caps from the score formula:

- score 0.1919853363: at most 288,327 bytes if distortion were zero;
- score 0.2053300290: at most 308,369 bytes if distortion were zero;
- score 0.10: 150,182 bytes;
- score 0.05: 75,091 bytes.

So the 152.2 MB Z8 archive needs roughly 500x-1000x byte collapse depending on
the distortion room reserved. The existing 4-pair coefficient deadzone and
relinearized smokes prove the materializer path, but they do not yet prove the
needed full-video floor.

## Full-video VJP and adjoint mechanics

The correct optimization object is the full contest objective over all pairs,
not minibatches:

S(A) = R(A) + 100 * D_seg(inflate(A)) + 5 * D_pose(inflate(A)).

For a transform coefficient c_i, the local deletion or quantization priority is
the adjoint-pushed score gradient:

g_i = <dS/dx, dx/dc_i>

with the caveat that the scorer losses are nonsmooth and the archive transform
contains discrete coding decisions. The valid engineering pattern is therefore:

- compute full-video VJP shards against one archive SHA;
- reduce shards once, without optimizer updates between shards;
- push pixel saliency through the DWT/direct-transform adjoint;
- mutate a trust-region candidate archive;
- re-inflate and re-score/advisory-check;
- relinearize after every accepted mutation.

The current Z8 guard design is mathematically aligned with this: minibatch
gradients may rank probes, but cannot authorize mutation, promotion, or score.

## Direct-transform Mamba

The most promising Mamba variant is not "store a Mamba state" in the archive.
It is a causal or bidirectional-context predictor over transform-domain tokens:

- tokens: pair, frame side, level, subband, channel, block, coefficient group;
- inputs: parent LL context, same-level neighbors, motion/context summaries,
  previous pair state, Dreamer/RSSM category, INR base residual tag;
- outputs: conditional mean/scale, zero/deadzone probability, entropy context,
  or mode logits for the coefficient block.

The charged packet must contain only model weights that are actually used by the
runtime plus entropy-coded residuals/categories. The predicted hidden state is a
decoder-side computation, not a sidecar. This converts Mamba from an advisory
temporal model into a real conditional entropy model:

L = code_length(q(c - mu_theta(context))) + bytes(theta) + distortion_penalty.

The shared Mamba-2 SSD helper is the right recurrence substrate for this because
it already exposes externalized state and tri-backend semantics. The missing
piece is an archive-consumed fixed runtime contract for coefficient prediction,
not another MLX-only trainer.

## Dreamer/RSSM role

Dreamer/RSSM is most useful as a discrete allocator and codebook selector:

- block mode: zero, scalar quant, vector quant, INR residual, protected;
- precision bucket: step size or deadzone radius;
- temporal state category: which Mamba context family to use;
- semantic protection class: P18/P19 sensitive, low-risk texture, occlusion,
  pose-joint region, boundary region.

The categorical Gumbel/STE, unimix, and annealing machinery is a good training
surface, but the export must commit exact categories and codebooks. A Dreamer
posterior in MLX is not an archive unless those categories are packed and the
runtime consumes them during inflate.

## Stack-of-stacks and INR synergy

The validated hierarchical predictive-coding stack implies a hybrid route:

1. INR/HNeRV/RNeRV style base renderer carries the low-frequency and semantic
   content at roughly frontier-class distortion;
2. Z8 direct-transform residual stream carries the scorer-relevant correction;
3. Mamba SSD predicts residual coefficient distributions across time/subbands;
4. Dreamer categories allocate block mode and precision;
5. full-video P18/P19 adjoint chooses what survives quantization;
6. exact archive proof decides whether the candidate is real.

This is more plausible than making the existing near-lossless wavelet blob
small enough alone. The current Z8 archive stores too much raw reconstruction
mass. A frontier candidate needs an INR base plus a small, scorer-aware residual
sidecar, not full-resolution coefficient custody.

## Applicability to Z8 joint P18/P19

Applicable:

- full-video P18/P19 VJP should drive coefficient protection and deletion;
- the existing Z8 coefficient waterfill materializer is the right first
  executable shell;
- Mamba/Dreamer can improve the implicit allocator and entropy model after the
  full-video surface exists;
- direct-transform coding can turn the wavelet blob from raw storage into a
  compressed residual stream.

Not applicable as authority:

- MLX-local loss, local advisory score, latent parity, or hidden-state mutation;
- any Mamba/Dreamer state not consumed by runtime inflate;
- CPU/CUDA or MLX axis conversion by inference;
- deadzone smoke ratios extrapolated to 600-pair authority without replay.

## Next implementation tasks

1. Build `Z8-FVJP-600`: full-video P18/P19 surface acquisition for the current
   600-pair Z8 archive, pinned to archive SHA, with shard manifests, exact
   reduction, DWT-adjoint consistency tests, and a stale-surface mutation guard.

2. Build `Z8-DirectTransform-Entropy-MVP`: parse the wavelet blob into
   subband/block tokens, run deterministic deadzone plus entropy coding
   variants, and report exact bytes by section. Start with zero/deadzone,
   scalar quant, brotli/zstd/rANS/range baselines before learned models.

3. Build `MambaCoeffPred-MVP`: train/evaluate a Mamba-2 SSD coefficient
   predictor that emits conditional residual parameters and an entropy context.
   Its acceptance metric is actual coded bytes plus replay distortion, not
   reconstruction loss alone.

4. Build `DreamerAllocator-MVP`: categorical block allocator over mode and
   precision buckets, trained with rate plus full-video P18/P19 distortion, then
   exported as committed categories/codebooks consumed by inflate.

5. Build `INR-Z8-Residual-Hybrid`: use an INR/HNeRV-family base packet for
   low-frequency semantics, then let Z8 direct-transform residual coding spend
   only on full-video P18/P19-sensitive errors. Target residual sidecar budgets
   in the 20k-100k byte range after the base packet.

6. Add proof gates before any score language: section mutation proof,
   full-frame inflate output comparison, runtime-tree SHA, archive SHA, exact
   CPU/CUDA payload, and explicit axis tags.

## Primary files inspected

- `src/tac/substrates/z8_hierarchical_predictive_coding/full_video_vjp_acquisition.py`
- `tools/build_z8_full_video_vjp_surface_bundle.py`
- `src/tac/optimization/joint_p18_p19_waterfill.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/joint_p18_p19_deadzone_rate_attack.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/joint_variational_driver.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/joint_coefficient_waterfill.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/archive.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/runtime_payload_bridge.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/mamba2_adapter.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/mlx_renderer.py`
- `src/tac/substrates/time_traveler_l5_z7_mamba2/architecture.py`
- `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_module.py`
- `src/tac/optimization/mamba2_predictor.py`
- `src/tac/substrates/_shared/mamba2_ssd/numpy_backend.py`
- `src/tac/substrates/_shared/mamba2_ssd/pytorch_backend.py`
- `src/tac/substrates/_shared/mamba2_ssd/mlx_backend.py`
- `src/tac/substrates/dreamer_v3_rssm/module.py`
- `src/tac/substrates/dreamer_v3_rssm/archive.py`
- `src/tac/substrates/dreamer_v3_rssm/inflate.py`
- `experiments/train_substrate_dreamer_v3_rssm.py`
- `src/tac/substrates/time_traveler_l5_z6/architecture.py`
- `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py`
- `src/tac/substrates/z6_v2_cargo_cult_unwind/architecture.py`
- `src/tac/substrates/z6_v2_cargo_cult_unwind/mlx_renderer.py`
- `src/tac/substrates/time_traveler_l5_z4/architecture.py`
- `src/tac/substrates/z4_cooperative_receiver_loss/architecture.py`

## Primary research/artifact files inspected

- `.omx/research/codex_findings_z8_full_video_surface_guard_20260531T171138Z_codex.md`
- `.omx/research/codex_findings_z8_relinearized_joint_coefficient_search_20260531T170553Z_codex.md`
- `.omx/research/codex_findings_z8_joint_p18_p19_executable_coeff_waterfill_20260531T170800Z_codex.md`
- `.omx/research/joint_p18_p19_gradient_waterfill_solver_architecture_20260531.md`
- `.omx/research/codex_findings_z8_joint_p18_p19_section_proof_hardening_20260531T163417Z_codex.md`
- `.omx/research/codex_findings_predictive_stack_argmax_hinge_default_20260531T141917Z_codex.md`
- `.omx/research/cross_z_stack_pixel_consumption_audit_20260531.md`
- `.omx/research/codex_findings_z8_wyner_ziv_pixel_driver_proof_closure_20260531T1600Z_codex.md`
- `.omx/research/codex_findings_z8_pixel_driver_and_segnet_grid_premise_20260531T153038Z_codex.md`
- `.omx/research/codex_session_summary_20260531T0215Z_codex.md`
- `.omx/research/codex_findings_mlx_archive_emitter_contract_closure_20260531T094918Z_codex.md`
- `reports/latest.md`
- `experiments/results/z8_600pair_byte_closed_contest_score_advisory/result.json`
- `experiments/results/z8_600pair_byte_closed_contest_score_advisory/byte_closed_archive/z8_hpc1_runtime_payload_bridge_report.json`
- `.omx/research/z8_hier_pc_full_stack_longrun_20260531/run/training_artifact.json`
- `.omx/research/z8_joint_p18_p19_full_video_relinearized_search_smoke_20260531T171138Z/candidate/z8_joint_p18_p19_relinearized_search_manifest.json`
- `.omx/research/z8_joint_p18_p19_deadzone_smoke_20260531T1715Z/candidate/z8_joint_p18_p19_deadzone_manifest.json`

