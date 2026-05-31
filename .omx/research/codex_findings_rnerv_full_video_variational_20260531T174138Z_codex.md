<!-- SPDX-License-Identifier: MIT -->
# Codex Findings: RNeRV/full-video variational implications

- Timestamp UTC: 2026-05-31T17:41:38Z
- Lane: `lane_codex_rnerv_full_video_variational_research_20260531`
- Scope: research/explorer design only; no source edits.
- Authority: research memo only. `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.
- Axis discipline: MLX-local and macOS advisory evidence are planning signals
  only. Contest score authority still requires byte-closed archive/runtime
  plus paired contest CPU/CUDA auth eval.

## Executive Verdict

RNeRV/NeRV-family long training is relevant, but the next useful Pact move is
not "add another INR substrate." The high-EV integration is to treat
RNeRV-style recurrence as a compressed temporal prior over the existing
per-pair latent tables and Z8 top-state/predictive-code variables, then test
whether the same scorer-aligned output can be represented with fewer bytes.

The first executable experiment should therefore be `RNeRV-lite as latent
generator`: train a recurrent/FFNeRV-style generator that materializes the
same `(num_pairs, latent_dim)` table currently consumed by PACT-NeRV/PR95-style
decoders, export those latents through the already byte-closed archive path,
and compare full-video component distances plus archive bytes against the
independent-latent baseline. This avoids a new runtime grammar on day one and
directly tests whether recurrence buys rate without breaking custody.

## Sources Inspected

Repo code and artifacts:

- `AGENTS.md`
- `CLAUDE.md`
- `PROGRAM.md`
- `src/tac/auth_eval_schema.py`
- `upstream/modules.py`
- `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
- `src/tac/local_acceleration/pr95_hnerv_mlx_stage_losses.py`
- `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
- `tools/export_z8_hier_pc_mlx_to_pytorch_state_dict.py`
- `experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py`
- `src/tac/substrates/pact_nerv_selector_v3/mlx_renderer.py`
- `src/tac/substrates/_shared/mlx_score_aware/harness.py`
- `src/tac/substrates/_shared/mlx_score_aware/loss.py`
- `src/tac/substrates/_shared/pact_nerv_full_main.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/full_video_vjp_acquisition.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/joint_coefficient_waterfill.py`
- `src/tac/substrates/z8_hierarchical_predictive_coding/runtime_payload_bridge.py`
- `.omx/research/pact_nerv_selector_v3_hinton_distill_600pair_extended_5000ep_landed_20260528.md`
- `.omx/research/pr95_mlx_byte_closed_contest_archive_export_landed_20260525.md`
- `.omx/research/codex_findings_z8_full_video_surface_guard_20260531T171138Z_codex.md`
- `.omx/research/codex_findings_z8_relinearized_joint_coefficient_search_20260531T170553Z_codex.md`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/lane_registry.json`

External primary sources:

- NeRV: https://arxiv.org/abs/2110.13903
- HNeRV: https://arxiv.org/abs/2304.02633
- RNeRV/VINRB paper: https://arxiv.org/abs/2506.24127
- VINRB code: https://github.com/mgwillia/vinrb
- VINRB RNeRV override example:
  https://raw.githubusercontent.com/mgwillia/vinrb/main/configs/overrides/rnerv-1_5.json
- VINRB encoder/block surfaces:
  https://raw.githubusercontent.com/mgwillia/vinrb/main/video_encoders/nerv_encoder.py
  and
  https://raw.githubusercontent.com/mgwillia/vinrb/main/models/layers/nerv_block.py

## Contest Objective, Written As The Variational Action

The official score helper is:

```text
S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489
```

The scorer implementation makes two details non-optional:

- SegNet preprocess uses only the last frame of each pair, then computes mean
  argmax disagreement.
- PoseNet preprocess consumes the pair as YUV6 and computes MSE on the first
  half of the pose head.

For a decoded pair sequence `Y_phi(i) = (Y0_i, Y1_i)` and source sequence
`X_i = (X0_i, X1_i)`, the contest variational problem is:

```text
min over phi, q, r:
  100 * E_i,p [ 1[argmax Seg(Y1_i)_p != argmax Seg(X1_i)_p] ]
  + sqrt(10 * E_i [ ||Pose(Y0_i,Y1_i)[0:6] - Pose(X0_i,X1_i)[0:6]||_2^2 ])
  + (25 / 37_545_489) * B(q, r)

subject to:
  q decodes through a contest-compliant inflate runtime,
  no scorer imports or external state in inflate,
  full 600-pair coverage for any budget-spending surface,
  CPU/CUDA auth eval before score authority.
```

Here `phi` is the realized decoder: global weights, recurrent state dynamics,
per-pair latents, wavelet residuals, selectors, quantization tables, and any
portable runtime constants. `q` is the byte-coded packet; `r` is the runtime
grammar. This is not an RGB-MSE problem, and not a PSNR problem.

The rate multiplier is:

```text
lambda_B = 25 / 37_545_489 = 0.000000665859 score points per byte
```

Useful byte anchors:

```text
  7,700 bytes   -> 0.005127 score
100,000 bytes   -> 0.066586 score
150,000 bytes   -> 0.099879 score
178,000 bytes   -> 0.118523 score
185,578 bytes   -> 0.123569 score
230,345 bytes   -> 0.153377 score
345,646 bytes   -> 0.230151 score
```

Therefore, a PR95-class 230 KB model only leaves about `0.0366` distortion
score if the target is `S < 0.19`. A 178 KB model leaves about `0.0715`.
This is why INRs must be judged by component scorer response and byte-closed
archive size, not by visual quality.

## Local Differential Form

For local optimization, replace the discontinuous SegNet argmax term with a
fail-closed surrogate, not a generic KL:

```text
m_i,p = Seg(Y1_i)_{c*_i,p} - max_{c != c*_i,p} Seg(Y1_i)_c
c*_i,p = argmax Seg(X1_i)_p
L_seg_surrogate = E_i,p [ w_i,p * softplus((margin - m_i,p) / tau) ]
```

The existing `boundary_argmax_hinge` and target-class/boundary-weighted
distillation direction is the right local shape because score changes only
when argmax decisions flip. Pixels far from a decision boundary have low
marginal utility even if their RGB error is visible.

For pose, if `D_pose = E ||r_pose||^2`, then:

```text
d/dD_pose sqrt(10 * D_pose) = 5 / sqrt(10 * D_pose)
```

This term becomes large near zero, so experiments need an epsilon-clipped or
trust-region version when ranking atoms. Otherwise the optimizer can hallucinate
infinite pose marginal utility from numerical noise.

For any atom group `g` with byte cost `DeltaB_g`:

```text
DeltaS_g ~= 100 * Delta d_seg_g
          + (5 / sqrt(10 * d_pose_current)) * Delta d_pose_g
          + lambda_B * DeltaB_g
```

Keep or add the atom only if expected `DeltaS_g < 0`; remove or quantize it if
expected `DeltaS_g > 0`. At optimum, every active atom class has equal
score-reduction per byte:

```text
-100 * d(d_seg)/dB
- (5 / sqrt(10 * d_pose)) * d(d_pose)/dB
= lambda_B
```

This is the KKT condition the floor estimator should fit against.

## How RNeRV Plugs Into The Full-Video Solve

Current relevant surfaces already imply the correct insertion point:

- PACT-NeRV/PR95-style MLX renderers have a shared global decoder and a
  trainable per-pair latent table `(num_pairs, latent_dim)`.
- PR95 packaging already understands decoder bytes plus latents bytes in a
  byte-closed archive grammar.
- Z8 already has full-video VJP acquisition, archive-pinned surfaces, and a
  coefficient water-fill materializer that treats budget-spending updates as
  full-video actions.
- Z8 runtime bridge already decodes Wyner-Ziv/Mamba top states and projects
  them into frame-1 top-LL, which is exactly the kind of receiver-side temporal
  state RNeRV should eventually become.

The right model family is:

```text
h_i = F_alpha(h_{i-1}, t_i, optional ego/pose/pair features)
z_i = G_beta(h_i) + eps_i
(Y0_i, Y1_i) = D_theta(z_i)
```

where:

- `theta` are global decoder weights.
- `alpha,beta` are the recurrent/FFNeRV temporal prior.
- `eps_i` is a sparse residual latent correction, entropy-coded by temporal
  delta/range coding.
- `D_theta` can initially be the existing PACT-NeRV/PR95 decoder family.
- Later, `F_alpha` can replace or augment Z8 top-state dynamics.

The first implementation should not ship a new inflate grammar. It should train
`F_alpha,G_beta,theta,eps` in MLX, then materialize the resulting `z_i` table
into the existing archive path. The A/B question is simple:

```text
Does recurrently generated z_i + compressed residual eps_i
match independent latent component distances at lower latent bytes?
```

If yes, only then promote recurrence into the portable inflate runtime, because
the byte win will justify the new grammar and parity work. If no, RNeRV remains
a training prior and not a runtime primitive.

## Theoretical Score-Floor Estimator

For each representation family `f`, estimate the lower envelope:

```text
S_f(B) = 100 * d_seg_f(B)
       + sqrt(10 * d_pose_f(B))
       + lambda_B * B

S_floor_f = min_B S_f(B)
```

The distortion curves must be fit from full-video component response, not RGB
MSE. The minimum viable estimator needs these rows per candidate:

```text
{
  archive_bytes,
  decoder_weight_bytes,
  latent_bytes,
  residual_bytes,
  runtime_tree_sha256,
  archive_sha256,
  full_video_pair_count,
  d_seg,
  d_pose,
  seg_boundary_flip_count,
  pose_first6_mse_by_dim,
  local_axis,
  score_claim=false
}
```

The first empirical fit can be a constrained two-component envelope:

```text
d_seg(B)  = d_seg_inf  + a_seg  * exp(-k_seg  * B_eff_seg)
d_pose(B) = d_pose_inf + a_pose * exp(-k_pose * B_eff_pose)
```

with separate byte buckets:

```text
B = B_global_weights + B_temporal_prior + B_latent_residual + B_wavelet_residual
```

The floor estimate becomes useful only when byte buckets are separable. A
global decoder byte amortizes over all frames; a latent byte is local to a pair;
a wavelet/residual byte is local to a region/frequency/frame. Those cannot be
merged into one "model size" scalar during allocation.

The contest-specific lower-bound sanity checks are:

- At `230,345` bytes, rate alone is `0.153377`; to beat `0.19`, total
  component distortion must be below `0.036623`.
- If `d_seg = 0.001`, the SegNet term already costs `0.1`, so a 230 KB archive
  cannot beat `0.19` no matter how low pose is.
- If `d_pose = 1e-4`, the pose term costs `0.031623`.
- The real floor requires both very small SegNet argmax disagreement and very
  small first-six PoseNet drift. PSNR SOTA is only relevant insofar as it
  predicts those two component responses.

## First Executable Experiments

### E0: Full-video component inventory before new code

Use current byte-closed or archive-bound candidates and produce full-video
component rows:

```text
input: current PACT-NeRV/PR95/Z8 archives or MLX training artifacts
output: JSONL rows with archive bytes, d_seg, d_pose, rate term, axis tags
```

Refuse pair-broadcast surfaces for any budget-spending conclusion. This reuses
the Z8 full-video surface rule: full pair-grid coverage or proposal-only.

### E1: RNeRV-lite as latent generator, no new archive grammar

Implement a new local MLX experiment that wraps the existing
`PactNervSelectorV3SubstrateMLX` decoder but replaces independent trainable
latents with:

```text
h_i = GRU_or_MLP(t_i, h_{i-1})
z_i = Linear(h_i) + eps_i
```

Then export/materialize the final `z_i` table into the existing PyTorch/archive
path. This tests the rate hypothesis without adding runtime risk.

Measurements:

- same full 600 pairs;
- same epoch/time budget as SELECTOR-V3 2000/5000 epoch anchors;
- component losses split into SegNet-only, PoseNet-only, combined;
- latent residual entropy and compressed latent bytes;
- MLX axis marked `score_claim=false`.

Acceptance condition:

```text
component score no worse than independent latents within tolerance
AND latent/residual bytes lower after the same archive coder
AND export/parity/readiness blockers are explicit
```

### E2: Component-axis sweep on existing PACT-NeRV anchors

Before a new architecture consumes time, replay the current selector-v3 style
training under three loss modes:

- SegNet boundary/argmax only;
- PoseNet first-six/Mahalanobis only;
- combined scorer-shaped loss.

This tells RNeRV whether its recurrence should spend capacity on frame-1
segmentation boundaries, pairwise pose dynamics, or both. It also identifies
whether the 5000 epoch plateau is a model-capacity ceiling or a mixed-loss
interference ceiling.

### E3: KKT water-fill across atom classes

Extend the Z8 full-video VJP/water-fill surface into a generic atom table:

```text
atom_class in {
  global_decoder_weight_group,
  recurrent_temporal_prior_weight_group,
  per_pair_latent_residual,
  z8_top_state,
  wavelet_detail_coeff,
  selector_side_channel
}
```

For each atom class record:

```text
Delta d_seg, Delta d_pose, Delta bytes, DeltaS, linearization_archive_sha
```

Then apply the KKT threshold `DeltaS < 0`. This makes RNeRV compete against
wavelet residuals and selector side channels rather than being evaluated in a
separate PSNR silo.

### E4: Byte-closed export smoke

Once E1 has a candidate, run the existing export/package/gate path:

- MLX state dict to PyTorch state dict;
- materialized latent table to archive grammar;
- portable inflate proof;
- full-frame inflate parity where applicable;
- advisory CPU/component eval only until paired auth eval.

No score language before this passes.

### E5: Runtime recurrence only after byte win

If E1 shows recurring `z_i` reduces latent bytes, add a portable runtime grammar
where the archive stores `alpha,beta,h_0` plus sparse corrections instead of
the full latent table. This is the second-stage version, not the first
experiment. It must include a manifest proving that recurrence bytes replace
latent bytes rather than adding a decorative model on top of them.

## Anti-Patterns And Custody Traps

1. **PSNR/LPIPS/visual SOTA as score authority.** NeRV/RNeRV literature metrics
   are useful priors, but the contest target is SegNet argmax, PoseNet first-six
   MSE, and archive bytes.
2. **RGB MSE as floor.** Current PR95 MLX long-training explicitly labels RGB
   MSE as a local MVP, not scorer-faithful.
3. **MLX as contest axis.** MLX rows can select follow-up and paid dispatch
   candidates; they cannot claim score, rank, kill, or promote.
4. **Pair-local gradients as full-video actions.** Any water-fill, recurrence,
   or rate attack needs full 600-pair coverage before budget-spending authority.
5. **Training-only recurrence.** If recurrence is trained but the archive still
   stores every latent raw, the representation may improve conditioning but not
   rate. That is a training prior, not compression.
6. **Raw hidden-state sidecar.** Storing `h_i` for every pair can erase any RNeRV
   byte win. Store global dynamics plus compressed residuals, or materialize
   latents only for the first experiment.
7. **Ignoring frame asymmetry.** SegNet sees only pair frame 1; PoseNet sees the
   pair. A symmetric reconstruction loss wastes bits.
8. **Pose square-root singularity.** Marginal utility near zero must be
   clipped/trust-regioned or it will over-rank noise.
9. **Archive/runtime mismatch.** Source archive plus source runtime must match
   the claimed result; adapter changes are new packets.
10. **Scorer in inflate.** The runtime may decode frames only. Scorer imports
    belong in training/eval, never contest inflate.
11. **No byte bucket accounting.** Global weights, temporal prior, latent
    residuals, and wavelet residuals have different amortization. One model-size
    scalar hides the actual allocation problem.
12. **Hybrid-INR compression eval trap.** VINRB itself warns that some hybrid
    INR compression evals can be unreliable because zeroing/model-saving
    boundaries miss parts of the bitstream. In Pact terms: no custody, no
    authority.

## Concrete Next Implementation Tasks

1. Add a research-only `rnerv_lite_latent_generator` MLX experiment that reuses
   the existing PACT-NeRV selector-v3 decoder and only changes latent production.
2. Add a component-row emitter for full 600-pair candidate packets:
   `archive_bytes`, `latent_bytes`, `decoder_bytes`, `d_seg`, `d_pose`,
   `axis_tag`, `archive_sha256`, `runtime_tree_sha256`, blockers.
3. Add a floor-estimator script that fits the component lower envelope and
   reports the KKT byte threshold against `lambda_B=25/37_545_489`.
4. Run a matched-budget A/B:
   independent latents vs recurrent generated latents plus residuals at 600
   pairs, 2000 epochs, then 5000 only if the 2000 epoch row is not dominated.
5. Wire the accepted candidate through existing export/package/parity gates
   before any exact-eval dispatch planning.

## Bottom Line

RNeRV is likely useful here only if it reduces the entropy of the temporal
latent/state sequence at the same scorer component response. The cleanest Pact
experiment is to make recurrence compete for bytes inside the existing
full-video variational water-fill, with materialized latents first and portable
runtime recurrence second. That gives a falsifiable local MLX path, preserves
archive custody, and keeps the contest objective rather than literature PSNR as
the action being optimized.
